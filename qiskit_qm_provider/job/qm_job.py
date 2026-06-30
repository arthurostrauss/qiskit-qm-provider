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
    SimulationConfig,
    StreamingResultFetcher,
    QuantumMachinesManager,
)
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.pending_job import QmPendingJob

from qiskit_qm_provider.backend.qm_backend import QMBackend
from qiskit_qm_provider.backend.backend_utils import (
    validate_circuits,
    measurement_output_bit_sizes,
)
from .iqcc_job_mixin import IQCCJobMixin, result_handles_from_qm_job
from .stream_assembly import bit_array_from_stream

if TYPE_CHECKING:
    from iqcc_cloud_client.qmm_cloud import CloudJob, CloudQuantumMachine
    from iqcc_cloud_client import IQCC_Cloud


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
        print(generate_qua_script(job.program))

    Attributes:
        program: Compiled :class:`qm.Program` (or ``list[Program]`` when
            ``backend.max_circuits`` splits a large batch into chunks). Set at
            construction; safe to inspect before ``submit()``.
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
        result_function: Callable[[RunningQmJob], Result],
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
    ) -> Callable[[RunningQmJob | List[RunningQmJob] | CloudJob | Dict], Result]:
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
            qm_jobs: List[RunningQmJob],
        ) -> Result:
            try:
                from iqcc_cloud_client.qmm_cloud import CloudResultHandles  # type: ignore[import]
            except ImportError:
                CloudResultHandles = None

            result_handle_types = (StreamingResultFetcher,)
            if CloudResultHandles is not None:
                result_handle_types = (StreamingResultFetcher, CloudResultHandles)

            results_handles = [job.result_handles for job in qm_jobs]
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
                circuit_metadata = getattr(circuits[i], "metadata", {}) or {}
                experiment_result = ExperimentResult(
                    shots=num_shots,
                    success=True,
                    data=ExperimentResultData(
                        data.get_counts(),
                        memory=data.get_bitstrings() if memory else None,
                    ),
                    meas_level=meas_level,
                    meas_return=meas_return,
                    header={"metadata": circuit_metadata},
                    status=getattr(qm_jobs[0], "status", "done"),
                )
                experiment_data.append(experiment_result)

            result = Result(
                results=experiment_data if num_circuits > 1 else experiment_data[0],
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
        - and submission of either a local ``QMJob`` or cloud ``IQCCJob``.
        """
        try:
            from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager  # type: ignore[import]
        except ImportError:
            CloudQuantumMachinesManager = None
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

        # Decide between local QM job and IQCCCloud job
        qmm_types = (QuantumMachinesManager,)
        if CloudQuantumMachinesManager is not None:
            qmm_types = (QuantumMachinesManager, CloudQuantumMachinesManager)
        if isinstance(backend.qmm, qmm_types):
            job_cls: type[QMJob] = QMJob
        else:
            job_cls = IQCCJob

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
        mapping = {
            "unknown": JobStatus.ERROR,
            "pending": JobStatus.QUEUED,
            "running": JobStatus.RUNNING,
            "completed": JobStatus.DONE,
            "canceled": JobStatus.CANCELLED,
            "loading": JobStatus.VALIDATING,
            "error": JobStatus.ERROR,
        }
        statuses = [
            mapping.get(getattr(job, "status", "unknown"), JobStatus.ERROR)
            for job in self._qm_jobs
        ]
        for state in (
            JobStatus.ERROR,
            JobStatus.CANCELLED,
            JobStatus.VALIDATING,
            JobStatus.QUEUED,
            JobStatus.RUNNING,
        ):
            if state in statuses:
                return state
        return JobStatus.DONE

    def submit(self):
        """Execute or queue the QUA program on the Quantum Machine."""
        compiler_options = self.metadata.get("compiler_options", None)
        simulate = self.metadata.get("simulate", None)
        if isinstance(self.qm, QuantumMachine):
            kwargs = {
                "simulate": simulate,
                "compiler_options": compiler_options,
            }
        else:  # CloudQuantumMachine
            kwargs = {
                "terminal_output": True,
            }
            if "timeout" in self.metadata:
                kwargs["options"] = {"timeout": self.metadata["timeout"]}
        if isinstance(simulate, SimulationConfig):
            if len(self.programs) > 1:
                self._qm_jobs = [
                    self.qm.simulate(
                        prog, simulate=simulate, compiler_options=compiler_options
                    )
                    for prog in self.programs
                ]
            else:
                self._qm_jobs = [self.qm.simulate(
                    self.programs[0], simulate=simulate, compiler_options=compiler_options
                )]
            self._job_id = ",".join(getattr(j, "id", "") for j in self._qm_jobs)
        else:
            if len(self.programs) > 1:
                self._qm_jobs = [self.qm.queue.add(prog, **kwargs) for prog in self.programs]
            else:
                self._qm_jobs = [self.qm.execute(self.programs[0], **kwargs)]
            self._job_id = ",".join(
                getattr(j, "id", "") for j in self._qm_jobs
            ).strip(",")

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


class IQCCJob(IQCCJobMixin, QMJob):
    """Job handle for IQCC cloud execution via :class:`~qiskit_qm_provider.providers.IQCCProvider`.

    Submits programs through the IQCC cloud client. Job status is not available via
    :meth:`status`; use IQCC cloud APIs to poll execution instead.

    Inspect cloud-side logs and failures via :attr:`run_data`. When the remote runtime
    failed, :meth:`result` raises :class:`~qiskit_qm_provider.job.IQCCCloudExecutionError`
    with the cloud ``stderr`` instead of a misleading local stream error.
    """

    def __init__(
        self,
        backend: QMBackend,
        job_id: str,
        qm: IQCC_Cloud,
        program: Program,
        result_function: Callable[[RunningQmJob], Result],
        **kwargs,
    ):
        super().__init__(backend, job_id, qm, program, result_function, **kwargs)

    def status(self) -> JobStatus:
        raise NotImplementedError("IQCCJob does not support status method. Use IQCC_Cloud methods to check job status.")

    def submit(self):
        """Submit the job to the IQCC backend."""
        if self._qm_jobs is not None:
            raise RuntimeError("IQCC job has already been submitted")
        try:
            config = self.metadata["config"]
        except KeyError:
            raise ValueError("Job metadata must contain 'config' key for IQCC job submission")

        qm: IQCC_Cloud = self.qm
        timeout = self.metadata.get("timeout", None)

        self._qm_jobs = [qm.execute(
            self.programs[0],
            config,
            options={"timeout": timeout} if timeout is not None else {},
        )]
        self._job_id = getattr(self._qm_jobs[0], "id", "")
