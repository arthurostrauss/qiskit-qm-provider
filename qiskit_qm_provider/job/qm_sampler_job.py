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
from qiskit.primitives.containers import SamplerPubResult, DataBin, BitArray
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
from .qua_programs import plan_sampler_programs
from .qm_primitive_job import QMPrimitiveJob

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
        # Keep a bare Program (not a 1-element list) for the single-program fast path.
        self._program = programs[0] if len(programs) == 1 else programs
        # Locator: global pub index -> (chunk_program_index, local_pub_index)
        self._locator = {
            g: (c, l)
            for c, chunk in enumerate(self._chunk_layout)
            for l, g in enumerate(chunk)
        }

    def _result_function(self, qm_job: Union[RunningQmJob, List[QmPendingJob]]) -> PrimitiveResult[SamplerPubResult]:
        is_job_list = isinstance(qm_job, list)
        if is_job_list:
            results_handles = [job.result_handles for job in qm_job]
            for handle in results_handles:
                handle.wait_for_all_values()
        else:
            results_handles = qm_job.result_handles
            results_handles.wait_for_all_values()

        meas_level = self.metadata.get("meas_level")
        all_data = []
        for i, pub in enumerate(self._pubs):
            chunk_idx, local_idx = self._locator[i]
            handle = results_handles[chunk_idx] if is_job_list else results_handles
            qc_meas_data = {}
            for output_key, bit_width in measurement_output_bit_sizes(pub.circuit).items():
                raw = handle.get(f"{output_key}_{local_idx}").fetch_all()
                data = np.asarray(raw)
                bit_array = BitArray.from_samples(data.tolist(), bit_width).reshape(pub.shape + (pub.shots,))
                qc_meas_data[output_key] = bit_array

            sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
            all_data.append(sampler_data)

        return PrimitiveResult(all_data)

    def submit(self):
        """Submit the job to the backend.

        When the PUBs were split into multiple QUA programs (chunked execution),
        each program is queued sequentially on QOP.  Results from all chunks are
        transparently stitched back in :meth:`_result_function`.
        """
        if self._qm_job is not None:
            raise RuntimeError("QM job has already been submitted")
        compiler_options: Optional[CompilerOptionArguments] = self.metadata.get("compiler_options", None)
        simulate: Optional[SimulationConfig] = self.metadata.get("simulate", None)

        programs = self._program if isinstance(self._program, list) else [self._program]

        if simulate is not None and isinstance(self._backend.qmm, QuantumMachinesManager):
            # Simulation only supports a single program — use the first chunk.
            self._qm_job = self._backend.qmm.simulate(
                self._backend.qm_config,
                programs[0],
                simulate=simulate,
                compiler_options=compiler_options,
            )
            self._job_id = self._qm_job.id
        elif len(programs) == 1:
            self._qm_job = self._backend.qm.execute(
                programs[0], compiler_options=compiler_options
            )
            self._job_id = self._qm_job.id
            self._push_parameters(self._qm_job, self._chunk_layout[0])
        else:
            self._qm_job = []
            for chunk_idx, (prog, chunk) in enumerate(zip(programs, self._chunk_layout)):
                job = self._backend.qm.queue.add(prog, compiler_options=compiler_options)
                self._qm_job.append(job)
                self._push_parameters(job, chunk)
            self._job_id = ",".join(j.id for j in self._qm_job)

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
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._result_function(self._qm_job)


class IQCCSamplerJob(IQCCJobMixin, QMSamplerJob):
    """IQCC Primitive Job class for executing QUA programs from PUBs."""

    def submit(self):
        """Submit the job to the backend.

        When PUBs were split into multiple QUA programs (chunked execution), each
        program is submitted as a separate IQCC cloud job with its own sync hook
        written to a system temp file.  Results are stitched back transparently
        in :meth:`_result_function` using the locator built at construction time.
        """
        from .post_hook_sampler import generate_sync_hook_sampler

        if self._qm_job is not None:
            raise RuntimeError("IQCC QM job has already been submitted")

        programs = self._program if isinstance(self._program, list) else [self._program]
        timeout = self.metadata.get("timeout", None)
        jobs = []

        for prog, chunk in zip(programs, self._chunk_layout):
            chunk_pubs = [self._pubs[g] for g in chunk]
            chunk_tables = [self._param_tables[g] for g in chunk]

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

            jobs.append(self._backend.qm.execute(prog, options=options))

        if len(jobs) == 1:
            self._qm_job = jobs[0]
            self._job_id = getattr(self._qm_job, "id", "")
        else:
            self._qm_job = jobs
            self._job_id = ",".join(getattr(j, "id", "") for j in jobs)

    def _result_function(self, qm_job) -> PrimitiveResult[SamplerPubResult]:
        """Get the result from the IQCC QM job(s).

        Handles both single-job and multi-chunk (list of jobs) cases.
        Uses ``_locator`` to fetch each PUB's measurement data from the correct
        chunk handle under the correct local stream key.
        """
        is_job_list = isinstance(qm_job, list)
        if is_job_list:
            results_handles = [job.result_handles for job in qm_job]
            for handle in results_handles:
                handle.wait_for_all_values()
        else:
            results_handles = qm_job.result_handles
            results_handles.wait_for_all_values()

        all_data = []
        for i, pub in enumerate(self._pubs):
            chunk_idx, local_idx = self._locator[i]
            handle = results_handles[chunk_idx] if is_job_list else results_handles
            qc_meas_data = {}
            for output_key, bit_width in measurement_output_bit_sizes(pub.circuit).items():
                raw = handle.get(f"{output_key}_{local_idx}").fetch_all()
                data = np.asarray(raw)
                bit_array = BitArray.from_samples(data.tolist(), bit_width).reshape(pub.shape + (pub.shots,))
                qc_meas_data[output_key] = bit_array

            sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
            all_data.append(sampler_data)

        return PrimitiveResult(all_data)

    def status(self) -> JobStatus:
        """Return the job status."""
        if self._qm_job is None:
            raise RuntimeError("IQCC QM job has not submitted yet")
        status = "completed"
        mapping = {
            "unknown": JobStatus.ERROR,
            "pending": JobStatus.QUEUED,
            "running": JobStatus.RUNNING,
            "completed": JobStatus.DONE,
            "canceled": JobStatus.CANCELLED,
            "loading": JobStatus.VALIDATING,
            "error": JobStatus.ERROR,
        }
        return mapping.get(status, JobStatus.ERROR)
