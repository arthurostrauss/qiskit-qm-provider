from __future__ import annotations

from typing import Optional, List, Callable, Union

from iqcc_cloud_client import IQCC_Cloud
from qiskit.providers.job import JobV1, JobStatus
from qiskit.result import Result
from qm import QuantumMachine, Program, SimulationConfig

from qiskit_qm_provider.backend.qm_backend import QMBackend
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.pending_job import QmPendingJob


class QMJob(JobV1):
    """QMJob class for Quantum Machines."""

    def __init__(
        self,
        backend: QMBackend,
        job_id: str,
        qm: Union[QuantumMachine, IQCC_Cloud],
        program: Program,
        result_function: Callable[[RunningQmJob], Result],
        **kwargs,
    ):

        JobV1.__init__(self, backend, job_id, **kwargs)
        self.qm = qm
        self._qm_job: Optional[RunningQmJob | QmPendingJob | List[QmPendingJob]] = None
        self.program = program
        self._result_function = result_function

    def status(self) -> JobStatus:
        """Return the job status."""
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        status = self._qm_job.status
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

    def submit(self):
        """Submit the job to the backend."""
        compiler_options = self.metadata.get("compiler_options", None)
        simulate = self.metadata.get("simulate", None)

        if isinstance(simulate, SimulationConfig):
            self._qm_job = self.qm.simulate(
                self.program, simulate=simulate, compiler_options=compiler_options
            )
        else:
            if isinstance(self.program, list):
                self._job_id = ""
                self._qm_job = []
                for prog in self.program:
                    self._qm_job.append(self.qm.queue.add(prog, compiler_options=compiler_options))
                self._job_id += ",".join([job.id for job in self._qm_job])
            else:
                self._qm_job = self.qm.execute(self.program, compiler_options=compiler_options)
                self._job_id = self._qm_job.id

    def cancel(self):
        """Cancel the job."""
        if self._qm_job is None:
            raise RuntimeError("QM job is not running")
        if isinstance(self._qm_job, list):
            for job in self._qm_job:
                job.cancel()
            return
        return self._qm_job.cancel()

    def result(self):
        """Get the job result."""
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")

        return self._result_function(self._qm_job)

    @property
    def qm_job(self) -> Optional[RunningQmJob | List[QmPendingJob | RunningQmJob]]:
        """Get the QM job."""
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._qm_job


class IQCCJob(QMJob):
    """IQCC Job class for Quantum Machines."""

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
        self._qm_job = None

    def status(self) -> JobStatus:
        raise NotImplementedError(
            "IQCCJob does not support status method. Use IQCC_Cloud methods to check job status."
        )

    def submit(self):
        """Submit the job to the IQCC backend."""
        if self._qm_job is not None:
            raise RuntimeError("IQCC job has already been submitted")
        try:
            config = self.metadata["config"]
        except KeyError:
            raise ValueError("Job metadata must contain 'config' key for IQCC job submission")

        qm: IQCC_Cloud = self.qm
        self._qm_job = qm.execute(self.program, config)
