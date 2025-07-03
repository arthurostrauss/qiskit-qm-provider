import numpy as np
from qiskit.circuit import Parameter
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.containers import DataBin, BitArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers import PubResult
from qiskit.providers import JobStatus
from qm import SimulationConfig, CompilerOptionArguments
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Union, List, Dict
from ..backend import QMBackend
from ..parameter_table import InputType, ParameterTable
from .qua_programs import sampler_program
from .qm_primitive_job import QMPrimitiveJob

class QMEstimatorJob(QMPrimitiveJob):
    def __init__(self, backend: QMBackend, pubs: List[EstimatorPub], input_type: InputType, **kwargs):
        super().__init__(backend, pubs, input_type, **kwargs)

    def _result_function(self, qm_job: Union[RunningQmJob, List[QmPendingJob]]) -> PrimitiveResult[PubResult]:
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
            if is_job_list:
                data = results_handle[i].get(f"counts_{i}").fetch_all()["value"]
            else:
                data = results_handle.get(f"counts_{i}").fetch_all()["value"]
            bit_array = BitArray.from_samples(data.tolist())
            bit_array = bit_array.reshape(pub.shape)


            
    def _run(self):
        pass

    def _parse_result(self):
        pass