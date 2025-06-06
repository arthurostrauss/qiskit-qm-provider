import numpy as np
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.primitives.containers import SamplerPubResult, DataBin, BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.providers import JobStatus
from qm import SimulationConfig, CompilerOptionArguments
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Union, List, Dict
from ..backend import QMBackend
from ..parameter_table import InputType
from .qua_programs import sampler_program


class QMPrimitiveJob(BasePrimitiveJob):
    """QM Primitive Job class for executing QUA programs from PUBs."""

    def __init__(self, backend: QMBackend, pubs: List[SamplerPub], input_type: InputType, **kwargs):
        super().__init__(job_id="pending", **kwargs)
        self._backend = backend
        self._pubs = pubs
        self._input_type = input_type
        self._qm_job: Optional[Union[RunningQmJob, QmPendingJob, List[QmPendingJob]]] = None

    def _result_function(
        self, qm_job: Union[RunningQmJob, List[QmPendingJob]]
    ) -> PrimitiveResult[SamplerPubResult]:

        is_job_list = isinstance(qm_job, list)
        if is_job_list:
            results_handle = [job.result_handles for job in qm_job]
            for handle in results_handle:
                handle.wait_for_all_values()
        else:
            results_handle = qm_job.result_handles
            results_handle.wait_for_all_values()

        all_data = []
        for i, pub in enumerate(self._pubs):
            qc_meas_data = {}
            for creg in pub.circuit.cregs:
                if is_job_list:
                    data = results_handle[i].get(f"{creg.name}_{i}").fetch_all()["value"]
                else:
                    data = results_handle.get(f"{creg.name}_{i}").fetch_all()["value"]
                meas_level = self.metadata.get("meas_level")
                if meas_level == "classified":
                    bit_array = BitArray.from_samples(data.tolist(), creg.size).reshape(pub.shape)
                    qc_meas_data[creg.name] = bit_array
                elif meas_level == "kerneled":
                    # TODO: Assume that buffering was done like (2, creg.size)
                    qc_meas_data[creg.name] = np.array(
                        [d[0] + 1j * d[1] for d in data], dtype=complex
                    ).reshape(pub.shape + (pub.shots, creg.size))
                else:
                    # TODO: Figure it out
                    qc_meas_data[creg.name] = np.array(
                        [d[0] + 1j * d[1] for d in data], dtype=complex
                    ).reshape(pub.shape + (pub.shots, creg.size))

            sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
            all_data.append(sampler_data)

        result = PrimitiveResult(all_data)
        return result

    def get_sampler_program(self):
        return sampler_program(self._backend, self._pubs, self._input_type)

    def submit(self):
        """Submit the job to the backend."""
        sampler_prog = sampler_program(self._backend, self._pubs, self._input_type, **self.metadata)
        if self._qm_job is not None:
            raise RuntimeError("QM job has already been submitted")
        compiler_options: Optional[CompilerOptionArguments] = self.metadata.get(
            "compiler_options", None
        )
        simulate: Optional[SimulationConfig] = self.metadata.get("simulate", None)
        if simulate is not None:
            self._qm_job = self._backend.qm.simulate(
                sampler_prog, simulate=simulate, compiler_options=compiler_options
            )
        else:
            self._qm_job = self._backend.qm.execute(sampler_prog, compiler_options=compiler_options)
            self._job_id = self._qm_job.id

    def result(self) -> PrimitiveResult[SamplerPubResult]:
        """Get the job result."""
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._result_function(self._qm_job)

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

    def done(self) -> bool:
        """Return whether the job has successfully run."""
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        """Return whether the job is actively running."""
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        """Return whether the job has been cancelled."""
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        """Return whether the job is in a final job state such as ``DONE`` or ``ERROR``."""
        return self.status() in [JobStatus.DONE, JobStatus.ERROR]

    def cancel(self):
        """Attempt to cancel the job."""
        if self._qm_job is None:
            raise RuntimeError("QM job is not running")
        return self._qm_job.cancel()

    @property
    def qm_job(self) -> Optional[Union[RunningQmJob, List[QmPendingJob]]]:
        """Return the QM job."""
        return self._qm_job

    @property
    def inputs(self) -> Dict:
        """Job input parameters.

        Returns:
            Input parameters used in this job.
        """

        return {
            "pubs": [(pub.circuit, pub.parameter_values, pub.shots) for pub in self._pubs],
            "input_type": self._input_type,
            "metadata": self.metadata,
        }


class IQCCPrimitiveJob(QMPrimitiveJob):
    """IQCC Primitive Job class for executing QUA programs from PUBs."""

    def submit(self):
        """Submit the job to the backend."""
        sampler_prog = sampler_program(self._backend, self._pubs, self._input_type)
        if self._qm_job is not None:
            raise RuntimeError("IQCC QM job has already been submitted")
        options = {"timeout": self.metadata.get("timeout", None)}
        self._qm_job = self._backend.qmm.execute(
            sampler_prog, self._backend.qm_config, options=options if options["timeout"] else {}
        )

    def _result_function(self, qm_job: Dict) -> PrimitiveResult[SamplerPubResult]:
        """Get the result from the IQCC QM job."""
        results_handle = qm_job["result"]
        all_data = []
        for i, pub in enumerate(self._pubs):
            qc_meas_data = {}
            for creg in pub.circuit.cregs:
                data = np.array(results_handle.get(f"{creg.name}_{i}")).flatten().tolist()
                bit_array = BitArray.from_samples(data, creg.size).reshape(pub.shape)
                qc_meas_data[creg.name] = bit_array

            sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
            all_data.append(sampler_data)

        result = PrimitiveResult(all_data)
        return result

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
