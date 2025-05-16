from __future__ import annotations

from typing import Optional, Dict, List, Callable
from collections import Counter

from qiskit.primitives import BitArray, SamplerPubResult, DataBin
from qiskit.providers.job import JobV1, JobStatus
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.result import Result, Counts
from qm import QuantumMachine, Program, SimulationConfig
from qm.grpc.frontend import SimulatedResponsePart

from .qm_backend import QMBackend
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.base_job import QmBaseJob
from .backend_utils import binary


class QMJob(JobV1):
    """QMJob class for Quantum Machines."""

    def __init__(
        self,
        backend: QMBackend,
        job_id: str,
        qm: QuantumMachine,
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
