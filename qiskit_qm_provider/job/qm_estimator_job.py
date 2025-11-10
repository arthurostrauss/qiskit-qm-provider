import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.base.base_primitive_job import ResultT
from qiskit.primitives.containers import DataBin, BitArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers import PubResult
from qm import SimulationConfig, CompilerOptionArguments
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Union, List, Dict, Tuple
from ..backend import QMBackend
from ..parameter_table import InputType, ParameterTable
from .qua_programs import estimator_program
from .qm_primitive_job import QMPrimitiveJob
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class _ExecutionPlan:
    """Holds the pre-computed execution plan for a single EstimatorPub."""

    # The tasks to be executed sequentially by the QUA program
    total_tasks: int

    # For each task, the index into the unique_params array
    task_param_indices: List[int]

    # For each task, the tuple of observable indices for the measurement basis
    task_obs_indices: List[Tuple[int, ...]]

    # The unique parameter value sets to be pushed
    unique_params: List[np.ndarray]

    # Metadata for post-processing to map flat results back to the original pub shape
    result_map: Dict[Tuple[int, Tuple[int, ...]], List[np.ndarray]] = field(default_factory=dict)
    shots_per_task: int = 0

    @classmethod
    def from_pub(cls, pub: EstimatorPub) -> "_ExecutionPlan":
        """
        Create an execution plan from a single EstimatorPub.

        Args:
            pub: The EstimatorPub to create the plan for.
        """

        # 1. Create an array of indices for the parameter values array.
        #    This is the key change to align with BackendEstimatorV2's logic.
        param_shape = pub.parameter_values.shape
        param_indices_array = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)

        # 2. Broadcast the parameter indices against the observables.
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices_array, pub.observables)

        # 3. Group observables by unique parameter index.
        #    The keys are now tuples of indices, e.g., (0,), (0, 1), etc.
        param_obs_map = defaultdict(set)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index_tuple = bc_param_ind[index]
            param_obs_map[param_index_tuple].update(bc_obs[index])

        # 4. Create the flat task lists for the QUA program.
        unique_param_indices = list(param_obs_map.keys())

        # Create a list of the actual unique parameter values by looking them up with their index.
        unique_params_values = [pub.parameter_values[p_idx].as_array() for p_idx in unique_param_indices]

        # Map the original index tuple to its new position in the unique list.
        param_idx_to_unique_idx_map = {p_idx_tuple: i for i, p_idx_tuple in enumerate(unique_param_indices)}

        task_param_indices = []
        task_obs_indices = []
        result_map = defaultdict(list)  # Metadata for post-processing

        for p_idx_tuple, observables in param_obs_map.items():
            # Get the index for the unique parameter set.
            unique_p_idx = param_idx_to_unique_idx_map[p_idx_tuple]

            # Group the collected observables into commuting measurement bases.
            # We assume observables in the list are SparsePauliOp objects.
            obs_pauli_list = PauliList(sorted(observables))
            commuting_groups_indices = observables_to_indices(obs_pauli_list)

            for obs_group_tuple in commuting_groups_indices:
                task_param_indices.append(unique_p_idx)
                task_obs_indices.append(obs_group_tuple)
                # Link this task back to the original experiments for result reconstruction.
                # This part may need further refinement based on the exact post-processing needs.
                result_map[(unique_p_idx, obs_group_tuple)].append(np.where(bc_param_ind == p_idx_tuple))

        return cls(
            total_tasks=len(task_param_indices),
            task_param_indices=task_param_indices,
            task_obs_indices=task_obs_indices,
            unique_params=unique_params_values,
            result_map=result_map,
            shots_per_task=int(1 / pub.precision**2),
        )


def observables_to_indices(
    observables: List[SparsePauliOp|Pauli|str] | SparsePauliOp | PauliList | Pauli | str,
):
    """
    Get single qubit indices of Pauli observables for the reward computation.

    Args:
        observables: Pauli observables to sample
    """
    if isinstance(observables, (str, Pauli)):
        observables = PauliList(Pauli(observables) if isinstance(observables, str) else observables)
    elif isinstance(observables, List) and all(isinstance(obs, (str, Pauli)) for obs in observables):
        observables = PauliList([Pauli(obs) if isinstance(obs, str) else obs for obs in observables])
    observable_indices = []
    observables_grouping = (
        observables.group_commuting(qubit_wise=True)
        if isinstance(observables, (SparsePauliOp, PauliList))
        else observables
    )
    for obs_group in observables_grouping:  # Get indices of Pauli observables
        current_indices = []
        paulis = obs_group.paulis if isinstance(obs_group, SparsePauliOp) else obs_group
        reference_pauli = Pauli((np.logical_or.reduce(paulis.z), np.logical_or.reduce(paulis.x)))
        for pauli_term in reversed(reference_pauli.to_label()):  # Get individual qubit indices for each Pauli term
            if pauli_term == "I" or pauli_term == "Z":
                current_indices.append(0)
            elif pauli_term == "X":
                current_indices.append(1)
            elif pauli_term == "Y":
                current_indices.append(2)
        observable_indices.append(tuple(current_indices))
    return observable_indices


class QMEstimatorJob(QMPrimitiveJob):
    def result(self) -> ResultT:
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._result_function(self._qm_job)

    def __init__(self, backend: QMBackend, pubs: List[EstimatorPub], input_type: InputType, **kwargs):
        super().__init__(backend, pubs, input_type, **kwargs)
        self._execution_plans: Optional[List[_ExecutionPlan]] = None

    def submit(self):
        """Submit the job to the backend after creating an efficient execution plan."""
        if self._qm_job is not None:
            raise RuntimeError("Job has already been submitted.")

        # 1. PRE-PROCESSING: Create an execution plan for each pub.
        self._execution_plans = [_ExecutionPlan.from_pub(pub) for pub in self._pubs]

        # 2. PROGRAM CREATION: Pass the plans to the program builder.
        estimator_prog = estimator_program(
            self._backend, self._pubs, self._input_type, execution_plans=self._execution_plans
        )
        self._program = estimator_prog
        compiler_options = self.metadata.get("compiler_options", None)

        # 3. EXECUTION: Start the QUA program on the OPX. It will wait for data.
        self._qm_job = self._backend.qm.execute(estimator_prog, compiler_options=compiler_options)
        self._job_id = self._qm_job.id

        # 4. DATA PUSHING: Loop through the planned tasks and push data to the running job.
        for i, pub in enumerate(self._pubs):
            plan = self._execution_plans[i]

            # Create ParameterTable objects once before the loop
            param_table = None
            if pub.circuit.parameters:
                param_table = ParameterTable.from_qiskit(
                    pub.circuit,
                    input_type=self._input_type,
                    filter_function=lambda p: isinstance(p, Parameter),
                )

            obs_vars = ParameterTable.from_qiskit(
                pub.circuit,
                input_type=self._input_type,
                filter_function=lambda p: pub.circuit.has_var(p) and "obs_" in p.name,
            )

            # This is the main loop that feeds the running QUA program
            for task_idx in range(plan.total_tasks):
                # Get the data for the current task from the plan
                param_idx = plan.task_param_indices[task_idx]
                param_values = plan.unique_params[param_idx]

                obs_indices_tuple = plan.task_obs_indices[task_idx]

                # Push parameter values if the circuit has them
                if param_table and len(param_values) > 0:
                    param_dict = {param.name: value for param, value in zip(param_table.parameters, param_values)}
                    param_table.push_to_opx(param_dict, self.qm_job, self._backend.qm)

                # Push observable indices for the measurement basis switch
                obs_indices_dict = {f"obs_{i}": val for i, val in enumerate(obs_indices_tuple)}
                obs_vars.push_to_opx(obs_indices_dict, self.qm_job, self._backend.qm)

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
            bit_array = BitArray.from_samples(data.tolist()).reshape(pub.shape)


    def _run(self):
        pass

    def _parse_result(self):
        pass
