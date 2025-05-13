from typing import Optional

from qiskit.providers.job import JobV1, JobStatus
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.result import Result
from qm import QuantumMachine, Program
from qm.grpc.frontend import SimulatedResponsePart

from .qm_backend import QMBackend
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.base_job import QmBaseJob

class QMJob(JobV1):
    """QMJob class for Quantum Machines."""

    def __init__(self, backend: QMBackend, 
                 job_id: str, 
                 qm: QuantumMachine,
                 program: Program,
                 run_input_length: int,
                 **kwargs):
        
        JobV1.__init__(self, backend, job_id, **kwargs)
        self.qm = qm
        self._qm_job: Optional[RunningQmJob] = None
        self.program = program
        self._run_input_length = run_input_length

    
    def status(self) -> JobStatus:
        """Return the job status."""
        if self._qm_job is None:
            raise RuntimeError('QM job has not submitted yet')
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
        self._qm_job = self.qm.execute(self.program,
                                       **self.metadata)
        self._job_id = self._qm_job.id
        
    def cancel(self):
        """Cancel the job."""
        if self._qm_job is None:
            raise RuntimeError('QM job is not running')
        return self._qm_job.cancel()
    
    def result(self):
        """Get the job result."""
        if self._qm_job is None:
            raise RuntimeError('QM job has not submitted yet')
        results_handle = self._qm_job.result_handles
        results_handle.wait_for_all_values()
        
        state_int = results_handle.get('all_measurements').fetch_all()["value"]
        
        
        
        
        
        # TODO: implement result fetching
