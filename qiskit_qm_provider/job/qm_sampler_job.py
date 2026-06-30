# Copyright 2026 Arthur Strauss
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QM sampler job: runs sampler PUBs as QUA programs and returns measurement counts.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations
from typing import Optional, Union, List, TYPE_CHECKING
import os
import inspect
import tempfile

import numpy as np

from qiskit.circuit import Parameter
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.containers import SamplerPubResult, DataBin
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.providers import JobStatus
from qiskit.result.models import MeasLevel

from qm import (
    SimulationConfig,
    CompilerOptionArguments,
    QuantumMachinesManager,
    generate_qua_script,
)
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.running_qm_job import RunningQmJob

from ..backend import QMBackend
from ..backend.backend_utils import measurement_output_bit_sizes, require_classified_meas_level
from ..parameter_table import InputType, ParameterPool, ParameterTable
from .iqcc_job_mixin import IQCCJobMixin
from .qua_programs import plan_sampler_programs, compute_locator
from .qm_primitive_job import QMPrimitiveJob
from .stream_assembly import bit_array_from_measurement_stream

if TYPE_CHECKING:
    from iqcc_cloud_client.qmm_cloud import CloudJob


class QMSamplerJob(QMPrimitiveJob):
    """Job handle for :class:`~qiskit_qm_provider.primitives.QMSamplerV2` execution.

    Builds a QUA sampler program from pubs and returns classified counts via
    :meth:`result`. See :attr:`program` for the compiled QUA source.
    """

    def __init__(
        self,
        backend: QMBackend,
        pubs: List[SamplerPub],
        input_type: InputType,
        **kwargs,
    ):
        """Create a sampler job.

        Args:
            backend: Backend that compiled the circuits.
            pubs: Coerced sampler pubs to execute.
            input_type: How circuit parameters are streamed to the OPX.
            **kwargs: Additional options forwarded to QUA program generation
                and backend execution (e.g. ``shots``, ``meas_level``).
        """
        super().__init__(backend, pubs, input_type, **kwargs)
        require_classified_meas_level(
            self.metadata.get("meas_level", MeasLevel.CLASSIFIED),
            context="QMSamplerJob",
        )
        ParameterPool.reset()
        self._param_tables = [
            ParameterTable.from_qiskit(
                pub.circuit,
                input_type=self._input_type,
                filter_function=lambda x: isinstance(x, Parameter),
                name=f"param_table_{i}",
            )
            for i, pub in enumerate(self._pubs)
        ]
        programs, self._chunk_layout = plan_sampler_programs(
            self._backend,
            self._pubs,
            self._param_tables,
            **self.metadata,
        )
        self._programs = programs
        self._locator = compute_locator(self._chunk_layout)

    def _result_function(self, qm_jobs: List[RunningQmJob]) -> PrimitiveResult[SamplerPubResult]:
        running_jobs = [
            j.wait_for_execution() if isinstance(j, QmPendingJob) else j
            for j in qm_jobs
        ]
        results_handles = [job.result_handles for job in running_jobs]
        for handle in results_handles:
            handle.wait_for_all_values()

        all_data = []
        for i, pub in enumerate(self._pubs):
            chunk_idx, local_idx = self._locator[i]
            handle = results_handles[chunk_idx]
            qc_meas_data = {}
            for output_key, bit_width in measurement_output_bit_sizes(pub.circuit).items():
                raw = handle.get(f"{output_key}_{local_idx}").fetch_all()
                qc_meas_data[output_key] = bit_array_from_measurement_stream(pub, raw, bit_width)

            sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
            all_data.append(sampler_data)

        return PrimitiveResult(all_data)

    def submit(self):
        """Submit the job to the backend.

        All QUA programs are first compiled via ``qm.compile()``, then added to
        the OPX queue via ``qm.queue.add_compiled()``.  Separating compilation
        from execution means the queue never stalls waiting for recompilation of
        later chunks — all programs are compiled upfront so the OPX can start
        executing them as soon as the queue is free.

        For simulation runs, programs are submitted directly to the simulator
        (no queue).  Results from all chunks are stitched back in
        :meth:`_result_function` using the locator built at construction time.
        """
        if self._qm_jobs is not None:
            raise RuntimeError("QM job has already been submitted")
        compiler_options: Optional[CompilerOptionArguments] = self.metadata.get("compiler_options", None)
        simulate: Optional[SimulationConfig] = self.metadata.get("simulate", None)

        programs = self._programs

        if simulate is not None and isinstance(self._backend.qmm, QuantumMachinesManager):
            self._qm_jobs = [
                self._backend.qmm.simulate(
                    self._backend.qm_config,
                    prog,
                    simulate=simulate,
                    compiler_options=compiler_options,
                )
                for prog in programs
            ]
            self._job_id = ",".join(getattr(j, "id", "") for j in self._qm_jobs)
        else:
            program_ids = [
                self._backend.qm.compile(prog, compiler_options=compiler_options)
                for prog in programs
            ]
            pending_jobs = [
                self._backend.qm.queue.add_compiled(pid) for pid in program_ids
            ]
            self._qm_jobs = pending_jobs
            self._job_id = ",".join(j.id for j in pending_jobs)
            for pending, chunk in zip(pending_jobs, self._chunk_layout):
                self._push_parameters(pending, chunk)

    def _push_parameters(self, qm_job, chunk: List[int]) -> None:
        """Stream circuit parameters to the OPX for the given chunk of pub indices."""
        for global_idx in chunk:
            pub = self._pubs[global_idx]
            param_table = self._param_tables[global_idx]
            if param_table is not None and param_table.input_type is not None:
                for parameters in pub.parameter_values.ravel().as_array():
                    param_dict = {
                        param.name: value
                        for param, value in zip(param_table.parameters, parameters)
                    }
                    param_table.push_to_opx(param_dict, qm_job, self._backend.qm)

    def result(self) -> PrimitiveResult[SamplerPubResult]:
        """Build and return classified measurement counts for all pubs.

        Returns:
            :class:`~qiskit.primitives.PrimitiveResult` with
            :class:`~qiskit.primitives.SamplerPubResult` entries.
        """
        if self._qm_jobs is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._result_function(self._qm_jobs)


class IQCCSamplerJob(IQCCJobMixin, QMSamplerJob):
    """IQCC cloud variant of :class:`QMSamplerJob`.

    Execution is **synchronous**: each :meth:`submit` call blocks until the
    remote OPX program completes.  ``CloudJob.status`` is therefore always
    ``"completed"``; real failure information lives in ``_run_data["stderr"]``
    (see :class:`~.IQCCJobMixin` and :meth:`~.IQCCJobMixin.status`).

    ``result()`` raises :class:`~.IQCCCloudExecutionError` before attempting to
    fetch streams when any chunk job's stderr contains a Python traceback.
    """

    def submit(self):
        """Submit all QUA programs to the IQCC cloud backend.

        For each chunk produced by :func:`~.plan_sampler_programs`, a separate
        cloud job is executed synchronously.  When the chunk's PUBs have
        parameterised circuits, a sync-hook script is written to a temporary
        file and passed to ``execute()``; the file is unlinked immediately
        after the call regardless of outcome.

        Results from all chunks are stitched back by :meth:`_result_function`
        using the locator built at construction time.
        """
        from .post_hook_sampler import generate_sync_hook_sampler

        if self._qm_jobs is not None:
            raise RuntimeError("IQCC QM job has already been submitted")

        programs = self._programs
        timeout = self.metadata.get("timeout", None)
        jobs = []

        for prog, chunk in zip(programs, self._chunk_layout):
            chunk_pubs = [self._pubs[g] for g in chunk]
            chunk_tables = [self._param_tables[g] for g in chunk]

            sync_hook_path = None
            if any(t is not None and t.input_type is not None for t in chunk_tables):
                sync_hook_code = generate_sync_hook_sampler(chunk_pubs, chunk_tables)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(sync_hook_code)
                    sync_hook_path = f.name
                options = {"sync_hook": sync_hook_path}
            else:
                options = {}

            if timeout is not None:
                options["timeout"] = timeout

            try:
                jobs.append(self._backend.qm.execute(prog, options=options))
            finally:
                if sync_hook_path is not None:
                    os.unlink(sync_hook_path)

        self._qm_jobs = jobs
        self._job_id = ",".join(getattr(j, "id", "") for j in jobs)

