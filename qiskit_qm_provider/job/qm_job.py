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

"""QMJob: Qiskit JobV1 implementation for Quantum Machines program execution.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from typing import Optional, List, Callable, Dict, Union, Any, TYPE_CHECKING

from copy import deepcopy

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import SamplerPubResult, DataBin
from qiskit.providers.job import JobV1, JobStatus
from qiskit.result import Result
from qiskit.result.models import (
    ExperimentResult,
    ExperimentResultData,
    MeasLevel,
    MeasReturnType,
)
from qm import (
    QuantumMachine,
    Program,
    StreamingResultFetcher,
)
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.pending_job import QmPendingJob

from qiskit_qm_provider.backend.qm_backend import QMBackend
from qiskit_qm_provider.backend.backend_utils import (
    validate_circuits,
    measurement_output_bit_sizes,
    experiment_result_header,
)
from .iqcc_job_mixin import IQCCJobMixin, result_handles_from_qm_job, aggregate_job_statuses
from .qm_execution_options import (
    ensure_job_running,
    is_cloud_quantum_machines_manager,
    join_job_ids,
    submit_qua_programs,
)
from .stream_assembly import bit_array_from_stream

if TYPE_CHECKING:
    from iqcc_cloud_client.qmm_cloud import CloudJob, CloudQuantumMachine


try:
    # Optional Qiskit Pulse import – mirrors backend behaviour but kept local
    from qiskit.pulse import DriveChannel  # type: ignore[unused-import]

    _QISKIT_PULSE_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without Pulse
    _QISKIT_PULSE_AVAILABLE = False


class QMJob(JobV1):
    """Qiskit job handle for QUA program execution on a QM backend.

    Returned by :meth:`~qiskit_qm_provider.backend.qm_backend.QMBackend.run`.
    Compile the generated QUA source with::

        from qm import generate_qua_script
        print(generate_qua_script(job.programs[0]))

    Compiled QUA programs are exposed on :attr:`programs` (always a list) and
    :attr:`program` (backward-compatible alias for the first program). Both are
    set at construction and safe to inspect before :meth:`submit`.

    Attributes:
        qm: :class:`qm.QuantumMachine` or cloud QM used for execution.
        backend: :class:`~qiskit_qm_provider.backend.qm_backend.QMBackend` that built
            the program.
        job_id: QM SDK job identifier (updated on ``submit()``).
        metadata: Run options dict (`compiler_options`, `simulate`, `timeout`, …).

    Use :attr:`qm_job` and :attr:`result_handles` for the live QM SDK object and
    measurement streams after submission.
    """

    def __init__(
        self,
        backend: QMBackend,
        job_id: str,
        qm: QuantumMachine | CloudQuantumMachine,
        program: Union[Program, List[Program]],
        result_function: Callable[[List[Union[RunningQmJob, QmPendingJob]]], Result],
        **kwargs,
    ):
        JobV1.__init__(self, backend, job_id, **kwargs)
        self.qm = qm
        self._qm_jobs: Optional[List[Union[RunningQmJob, QmPendingJob]]] = None
        self._programs: List[Program] = program if isinstance(program, list) else [program]
        self._result_function = result_function

    @property
    def programs(self) -> List[Program]:
        """Compiled QUA program(s) for this job; always a list."""
        return self._programs

    @property
    def program(self) -> Program:
        """Backward-compatible alias for :attr:`programs`\\ ``[0]``."""
        return self._programs[0]

    @property
    def qm_jobs(self) -> Optional[List[Union[RunningQmJob, QmPendingJob]]]:
        """Underlying QM SDK jobs after :meth:`submit` — always a list.

        Length 1 for single-program execution; one entry per chunk otherwise.
        """
        return self._qm_jobs

    def get_qm_job(self, idx: Optional[int] = None):
        """Return the QM SDK job at *idx* (default: first / only job).

        Convenience accessor equivalent to ``job.qm_jobs[idx]``.

        Raises:
            RuntimeError: If the job has not been submitted yet.
            IndexError: If *idx* is out of range.
        """
        if self._qm_jobs is None:
            raise RuntimeError("QM job has not been submitted yet")
        return self._qm_jobs[0 if idx is None else idx]

    def get_program(self, idx: Optional[int] = None) -> Program:
        """Return the compiled QUA program at *idx* (default: first / only program).

        Convenience accessor equivalent to ``job.programs[idx]``.

        Raises:
            IndexError: If *idx* is out of range.
        """
        return self._programs[0 if idx is None else idx]

    # ------------------------------------------------------------------
    # High-level constructors used by QMBackend.run
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result_function(
        backend: QMBackend,
        num_circuits: int,
        num_shots: int,
        circuits: List[QuantumCircuit],
        cregs_dicts: List[Dict[str, int]],
        meas_level: MeasLevel,
        meas_return: MeasReturnType,
        memory: bool,
        chunk_layout: Optional[List[List[int]]] = None,
    ) -> Callable[[List[Union[RunningQmJob, QmPendingJob]]], Result]:
        """Create a Result-building callback for standard circuit execution.

        This function encapsulates the data plumbing that was previously
        implemented as an inner closure in ``QMBackend.run``.

        ``chunk_layout`` maps each QUA program to the global circuit indices it
        holds (``chunk_layout[c]`` lists the global indices in program ``c``).
        Within a program, results are saved under stream key ``f"{creg}_{l}"``
        where ``l`` is the circuit's *local* index in that program. We invert
        this into a ``global index -> (chunk, local index)`` locator so each
        circuit's data is fetched from the right handle and key. When omitted
        (or a single program), the layout is a single chunk and the local index
        equals the global index, matching the historical behaviour.
        """
        from ..backend.backend_utils import require_classified_meas_level
        from .qua_programs import compute_locator

        require_classified_meas_level(meas_level, context="QMBackend.run()")

        if chunk_layout is None:
            chunk_layout = [list(range(num_circuits))]
        locator: Dict[int, tuple] = compute_locator(chunk_layout)

        def result_function(
            qm_jobs: List[Union[RunningQmJob, QmPendingJob]],
        ) -> Result:
            try:
                from iqcc_cloud_client.qmm_cloud import CloudResultHandles  # type: ignore[import]
            except ImportError:
                CloudResultHandles = None

            result_handle_types = (StreamingResultFetcher,)
            if CloudResultHandles is not None:
                result_handle_types = (StreamingResultFetcher, CloudResultHandles)

            running_jobs = [ensure_job_running(job) for job in qm_jobs]
            results_handles = [job.result_handles for job in running_jobs]
            for handle in results_handles:
                if isinstance(handle, result_handle_types):
                    handle.wait_for_all_values()

            all_data: List[SamplerPubResult] = []
            for i in range(num_circuits):
                qc_meas_data = {}
                chunk_idx, local_idx = locator[i]
                handle = results_handles[chunk_idx]
                for creg, creg_size in cregs_dicts[i].items():
                    key = f"{creg}_{local_idx}"
                    if isinstance(handle, result_handle_types):
                        raw = handle.get(key).fetch_all()
                    else:
                        raw = handle.get(key)
                    qc_meas_data[creg] = bit_array_from_stream(raw, creg_size, (num_shots,))

                sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
                all_data.append(sampler_data.join_data())

            experiment_data = []
            for i, data in enumerate(all_data):
                experiment_result = ExperimentResult(
                    shots=num_shots,
                    success=True,
                    data=ExperimentResultData(
                        counts=data.get_counts(),
                        memory=data.get_bitstrings() if memory else None,
                    ),
                    meas_level=meas_level,
                    meas_return=meas_return,
                    header=experiment_result_header(circuits[i]),
                    status=getattr(running_jobs[locator[i][0]], "status", "done"),
                )
                experiment_data.append(experiment_result)

            result = Result(
                results=experiment_data,
                backend_name=backend.name,
                job_id=",".join(getattr(j, "id", "") for j in qm_jobs),
                backend_version=2,
                qobj_id=None,
                success=True,
            )
            return result

        return result_function

    @classmethod
    def from_circuits(
        cls,
        backend: QMBackend,
        run_input: QuantumCircuit | List[QuantumCircuit],
        **options,
    ) -> "QMJob":
        """Factory that mirrors the original ``QMBackend.run`` logic.

        This method performs:

        - circuit validation and optional reset insertion,
        - target / calibration updates,
        - QUA program generation via ``plan_run_programs``,
        - result object construction from streamed data,
        - and submission via :meth:`QMJob.submit`
          (``CloudQuantumMachine.execute`` on IQCC cloud backends).
        """
        from .qua_programs import plan_run_programs, compute_locator

        # Merge explicit options into backend defaults (preserving current behaviour)
        options_ = deepcopy(backend.options.__dict__)
        options_.update(options)

        num_shots = options.get("shots", backend.options.shots)
        skip_reset = options.get("skip_reset", backend.options.skip_reset)
        memory = options.get("memory", backend.options.memory)
        meas_level = options.get("meas_level", backend.options.meas_level)
        meas_return = options.get("meas_return", backend.options.meas_return)

        if not isinstance(run_input, list):
            run_input = [run_input]

        new_circuits = validate_circuits(run_input, should_reset=not skip_reset, check_for_params=True)
        num_circuits = len(new_circuits)

        # Synchronize backend target and (optionally) pulse calibrations
        backend.update_target()
        if _QISKIT_PULSE_AVAILABLE:
            for qc in new_circuits:
                backend.update_calibrations(qc)

        # Build the QUA program(s) the QM will execute. Large batches are split
        # into several programs (<= backend.max_circuits circuits each) that are
        # queued sequentially; ``chunk_layout`` records which global circuit
        # indices live in each program so results can be stitched back together.
        programs, chunk_layout = plan_run_programs(backend, num_shots, new_circuits)
        qm = backend.qm

        job_id = "pending"
        cregs_dicts: List[Dict[str, int]] = [measurement_output_bit_sizes(qc) for qc in new_circuits]

        result_function = cls._build_result_function(
            backend=backend,
            num_circuits=num_circuits,
            num_shots=num_shots,
            circuits=new_circuits,
            cregs_dicts=cregs_dicts,
            meas_level=meas_level,
            meas_return=meas_return,
            memory=memory,
            chunk_layout=chunk_layout,
        )

        # Cloud backends (CloudQuantumMachinesManager → CloudQuantumMachine) use
        # the same QMJob submit path; CloudQMJob only adds IQCC stderr diagnostics.
        job_cls: type[QMJob] = (
            CloudQMJob if is_cloud_quantum_machines_manager(backend.qmm) else QMJob
        )

        job = job_cls(
            backend,
            job_id,
            qm,
            programs,
            result_function=result_function,
            **options_,
        )

        job.submit()
        # Reset the calibration mapping as in the original QMBackend.run implementation
        backend._calibration_operation_mapping_QUA = backend._operation_mapping_QUA.copy()  # type: ignore[attr-defined]
        return job

    def status(self) -> JobStatus:
        """Return Qiskit job status mapped from the underlying QM job(s).

        Aggregates across all chunk jobs: DONE only when every program is done.
        """
        if self._qm_jobs is None:
            raise RuntimeError("QM job has not submitted yet")
        return aggregate_job_statuses(self._qm_jobs)

    def submit(self):
        """Submit all QUA programs on the Quantum Machine.

        Cloud and simulation runs call ``qm.execute`` (with kwargs trimmed per
        QMM type; simulation uses ``simulate=SimulationConfig``). Real hardware
        enqueues every program via OPX1000 ``add_to_queue`` with OPX+
        ``queue.add`` as fallback — looping ``execute`` would clear the queue
        between chunks.
        """
        self._qm_jobs = submit_qua_programs(
            self.qm, self._backend.qmm, self.programs, self.metadata
        )
        self._job_id = join_job_ids(self._qm_jobs)

    def cancel(self):
        """Cancel all underlying QM job(s)."""
        if self._qm_jobs is None:
            raise RuntimeError("QM job is not running")
        for job in self._qm_jobs:
            job.cancel()

    def result(self):
        """Build and return a Qiskit :class:`~qiskit.result.Result` from QM streaming data."""
        if self._qm_jobs is None:
            raise RuntimeError("QM job has not submitted yet")

        return self._result_function(self._qm_jobs)

    @property
    def result_handles(self) -> Any:
        """QM SDK result stream handles after :meth:`submit`.

        Always a list — one ``result_handles`` per submitted job.
        Length 1 for non-chunked execution.  Raises if not yet submitted.
        """
        return result_handles_from_qm_job(self._qm_jobs)

    def get_result_handles(self, idx: Optional[int] = None):
        """Return the result-handles object at *idx* (default: first / only job).

        Convenience accessor equivalent to ``job.result_handles[idx]``.  Defaults
        to index 0, which is the correct handle for non-chunked execution.

        Raises:
            RuntimeError: If the job has not been submitted yet.
            IndexError: If *idx* is out of range.
        """
        return result_handles_from_qm_job(self._qm_jobs)[0 if idx is None else idx]


class CloudQMJob(IQCCJobMixin, QMJob):
    """:class:`QMJob` for IQCC :class:`~iqcc_cloud_client.qmm_cloud.CloudQuantumMachine` backends.

    Submission uses the same :func:`~.submit_qua_programs` path as :class:`QMJob`
    (``CloudQuantumMachine.execute``). This subclass only mixes in
    :class:`~.IQCCJobMixin` for cloud ``run_data`` / stderr diagnostics.

    Prefer constructing backends via :class:`~qiskit_qm_provider.providers.IQCCProvider`
    so ``machine.connect()`` yields a :class:`~iqcc_cloud_client.qmm_cloud.CloudQuantumMachinesManager`
    and ``backend.qm`` is a :class:`~iqcc_cloud_client.qmm_cloud.CloudQuantumMachine`.
    """
