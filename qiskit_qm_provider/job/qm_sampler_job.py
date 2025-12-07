from __future__ import annotations
import numpy as np
import os
import inspect
from qiskit.circuit import Parameter
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.containers import SamplerPubResult, DataBin, BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.providers import JobStatus

from qm import SimulationConfig, CompilerOptionArguments, QuantumMachinesManager
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Union, List, TYPE_CHECKING
from ..backend import QMBackend
from ..parameter_table import InputType, ParameterPool, ParameterTable
from .qua_programs import sampler_program
from .qm_primitive_job import QMPrimitiveJob
if TYPE_CHECKING:
    from iqcc_cloud_client.qmm_cloud import CloudJob

class QMSamplerJob(QMPrimitiveJob):
    """QM Primitive Job class for executing QUA programs from PUBs."""

    def __init__(self, backend: QMBackend, pubs: List[SamplerPub], input_type: InputType, **kwargs):
        super().__init__(backend, pubs, input_type, **kwargs)
        ParameterPool.reset()
        self._param_tables = [ParameterTable.from_qiskit(pub.circuit, input_type=self._input_type, filter_function=lambda x: isinstance(x, Parameter),
        name=f"param_table_{i}") for i, pub in enumerate(self._pubs)]
        self._program = sampler_program(self._backend, self._pubs, self._param_tables, **self.metadata)

    def _result_function(self, qm_job: Union[RunningQmJob, List[QmPendingJob]]) -> PrimitiveResult[SamplerPubResult]:
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
                    qc_meas_data[creg.name] = np.array([d[0] + 1j * d[1] for d in data], dtype=complex).reshape(
                        pub.shape + (pub.shots, creg.size)
                    )
                else:
                    # TODO: Figure it out
                    qc_meas_data[creg.name] = np.array([d[0] + 1j * d[1] for d in data], dtype=complex).reshape(
                        pub.shape + (pub.shots, creg.size)
                    )

            sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
            all_data.append(sampler_data)

        result = PrimitiveResult(all_data)
        return result

    def submit(self):
        """Submit the job to the backend."""
        sampler_prog = self._program
        if self._qm_job is not None:
            raise RuntimeError("QM job has already been submitted")
        compiler_options: Optional[CompilerOptionArguments] = self.metadata.get("compiler_options", None)
        simulate: Optional[SimulationConfig] = self.metadata.get("simulate", None)
        if simulate is not None and isinstance(self._backend.qmm, QuantumMachinesManager):
            self._qm_job = self._backend.qmm.simulate(self._backend.qm_config, sampler_prog, simulate=simulate, compiler_options=compiler_options)
            self._job_id = self._qm_job.id
        else:
            self._qm_job = self._backend.qm.execute(sampler_prog, compiler_options=compiler_options)
            self._job_id = self._qm_job.id
            for pub, param_table in zip(self._pubs, self._param_tables):
                if param_table is not None and param_table.input_type is not None:
                    for parameters in pub.parameter_values.ravel().as_array():
                        param_dict = {param.name: value for param, value in zip(param_table.parameters, parameters)}
                        param_table.push_to_opx(param_dict, self._qm_job, self._backend.qm)

    def result(self) -> PrimitiveResult[SamplerPubResult]:
        """Get the job result."""
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._result_function(self._qm_job)


class IQCCSamplerJob(QMSamplerJob):
    """IQCC Primitive Job class for executing QUA programs from PUBs."""

    def submit(self):
        """Submit the job to the backend."""
        from .post_hook_sampler import generate_sync_hook_sampler
        param_tables = self._param_tables
        sampler_prog = self._program
        if self._qm_job is not None:
            raise RuntimeError("IQCC QM job has already been submitted")
        if any(param_table is not None and param_table.input_type is not None for param_table in param_tables):
            sync_hook_code = generate_sync_hook_sampler(self._pubs, param_tables)
        else:
            sync_hook_code = None
        # Determine the calling context to get the script file path
        caller_frame = inspect.stack()[-1]
        main_script_path = caller_frame.filename
        main_script_dir = os.path.dirname(os.path.abspath(main_script_path))
        sync_hook_path = os.path.join(main_script_dir, "sync_hook_sampler.py")
        if sync_hook_code is not None:
            with open(sync_hook_path, "w") as f:
                f.write(sync_hook_code)
            options = {"sync_hook": sync_hook_path}
        else:
            options = {}
        self._qm_job = self._backend.qm.execute(
            sampler_prog, options=options
        )

    def _result_function(self, qm_job: CloudJob) -> PrimitiveResult[SamplerPubResult]:
        """Get the result from the IQCC QM job."""
        results_handle = qm_job.result_handles
        all_data = []
        for i, pub in enumerate(self._pubs):
            qc_meas_data = {}
            for creg in pub.circuit.cregs:
                data = np.array(results_handle.get(f"{creg.name}_{i}").fetch_all()).flatten().tolist()
                # BitArray.from_samples creates shape=() with num_shots=len(data)
                # To reshape to pub.shape, we need to include shots: pub.shape + (pub.shots,)
                # This makes the total size match self.size * self.num_shots
                bit_array = BitArray.from_samples(data, creg.size).reshape(pub.shape + (pub.shots,))
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
