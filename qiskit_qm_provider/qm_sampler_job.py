from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.primitives.containers import SamplerPubResult
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.running_qm_job import RunningQmJob


class QMPrimitiveJob(BasePrimitiveJob[PrimitiveResult[SamplerPubResult], JobStatus]):
    """QM Primitive Job class for executing QUA programs from PUBs."""

    def __init__(self, backend: QMBackend, pubs: List[SamplerPub], input_type: InputType):
        super().__init__(job_id="pending")
        self._backend = backend
        self._pubs = pubs
        self._input_type = input_type
        self._qm_job: Optional[Union[RunningQmJob, QmPendingJob, List[QmPendingJob]]] = None
        self._result_function = self._create_result_function()

    def _create_result_function(
        self,
    ) -> Callable[[Union[RunningQmJob, List[QmPendingJob]]], Result]:
        def result_function(qm_job: Union[RunningQmJob, List[QmPendingJob]]) -> Result:
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
                    bit_array = BitArray.from_samples(data, creg.size)
                    qc_meas_data[creg.name] = bit_array

                sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
                all_data.append(sampler_data.join_data())

            experiment_data = []
            for data in all_data:
                experiment_result = ExperimentResult(
                    shots=pub.shots,
                    success=True,
                    data=ExperimentResultData(
                        data.get_counts(),
                        memory=data.get_bitstrings() if self._backend.options.memory else None,
                    ),
                )
                experiment_data.append(experiment_result)

            result = Result(
                data=experiment_data if len(self._pubs) > 1 else experiment_data[0],
                header={"backend_name": self._backend.name, "job_id": qm_job.id},
            )
            return result

        return result_function

    def submit(self):
        """Submit the job to the backend."""
        circuits = [pub.circuit for pub in self._pubs]
        param_tables = [
            ParameterTable.from_qiskit(qc, input_type=self._input_type) for qc in circuits
        ]
        run_program = self._backend.get_run_program(self._pubs[0].shots, circuits)
        qm = self._backend.qm

        job = QMJob(
            self._backend,
            self._job_id,
            qm,
            run_program,
            result_function=self._result_function,
        )
        job.submit()
        self._qm_job = job.qm_job

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
