import numpy as np
from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.backend_estimator_v2 import _measurement_circuit
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
from ..primitives.qm_estimator import QMEstimatorOptions
from dataclasses import dataclass, field
from collections import defaultdict

def generate_qm_estimator_pubs(pubs: List[EstimatorPub], switch_obs_circuit: QuantumCircuit) -> List[EstimatorPub]:
    """
    Generate the QM estimator pubs from the given pubs and switch observation circuit.
    This pads the circuit with the switch observation circuit for each qubit.
    """
    new_pubs = []
    for pub in pubs:
        qc = pub.circuit.copy()
        creg = ClassicalRegister(pub.circuit.num_qubits, name="__c")
        qc.add_register(creg)
        for q in range(pub.circuit.num_qubits):
            qubit = qc.qubits[q]
            clbit = creg[q]
            qc.compose(switch_obs_circuit, [qubit], [clbit], inplace=True,
            var_remap={"obs": f"obs_{q}"})
        new_pub = EstimatorPub(qc, pub.observables, pub.parameter_values, pub.precision)
        new_pubs.append(new_pub)
    return new_pubs
@dataclass
class _ExecutionPlan:
    """Holds the pre-computed execution plan for a single EstimatorPub."""

    metadata: List[Dict]
    param_indices: np.ndarray
    obs_indices: List[List[Tuple[int, ...]]]

    @classmethod
    def from_pub(cls, pub: EstimatorPub, options: QMEstimatorOptions):
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)

        # 2. Broadcast the parameter indices against the observables.
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, observables)

        # 3. Group observables by unique parameter index.
        #    The keys are now tuples of indices, e.g., (0,), (0, 1), etc.
        param_obs_map = defaultdict(set)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            param_obs_map[param_index].update(bc_obs[index])

        metadata = []
        obs_indices_list = []
        for param_index, pauli_strings in param_obs_map.items():
            meas_paulis = PauliList(sorted(pauli_strings))
            obs_indices_list.append(observables_to_indices(meas_paulis))
            if options.abelian_grouping:
                for obs in meas_paulis.group_commuting(qubit_wise=True):
                    basis = Pauli((np.logical_or.reduce(obs.z), np.logical_or.reduce(obs.x)))
                    _, indices = _measurement_circuit(circuit.num_qubits, basis)
                    paulis = PauliList.from_symplectic(
                        obs.z[:, indices],
                        obs.x[:, indices],
                        obs.phase,
                    )
                    metadata.append({
                        "meas_paulis": paulis,
                        "param_index": param_index,
                        "orig_paulis": obs,
                    })
            else:
                for basis in meas_paulis:
                    _, indices = _measurement_circuit(circuit.num_qubits, basis)
                    obs = PauliList(basis)
                    paulis = PauliList.from_symplectic(
                        obs.z[:, indices],
                        obs.x[:, indices],
                        obs.phase,
                    )
                    metadata.append({
                        "meas_paulis": paulis,
                        "param_index": param_index,
                        "orig_paulis": obs,
                    })
        return cls(metadata, param_indices, obs_indices_list)

def observables_to_indices(
    observables: List[SparsePauliOp|Pauli|str] | SparsePauliOp | PauliList | Pauli | str,
    abelian_grouping: bool = True,
) -> List[Tuple[int, ...]]:
    """
    Get single qubit indices of Pauli observables for the reward computation.

    Args:
        observables: Pauli observables to sample

    Returns:
        List of tuples of single qubit indices for each qubit-wise commuting group of observables.
    """
    if isinstance(observables, (str, Pauli)):
        observables = PauliList(Pauli(observables) if isinstance(observables, str) else observables)
    elif isinstance(observables, List) and all(isinstance(obs, (str, Pauli)) for obs in observables):
        observables = PauliList([Pauli(obs) if isinstance(obs, str) else obs for obs in observables])
    observable_indices = []
    if abelian_grouping:
        observables_grouping = (
            observables.group_commuting(qubit_wise=True)
            if isinstance(observables, (SparsePauliOp, PauliList))
            else observables
        )
    else:
        observables_grouping = observables
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
        self._switch_obs_circuit: QuantumCircuit = kwargs["switch_obs_circuit"]
        
    def submit(self):
        """Submit the job to the backend after creating an efficient execution plan."""
        if self._qm_job is not None:
            raise RuntimeError("Job has already been submitted.")
        pubs = generate_qm_estimator_pubs(self._pubs, self._switch_obs_circuit)
        # 1. PRE-PROCESSING: Create an execution plan for each pub.
        self._execution_plans = [_ExecutionPlan.from_pub(pub, options=self._options) for pub in pubs]
        param_tables = [ParameterTable.from_qiskit(pub.circuit, input_type=self._input_type, filter_function=lambda p: isinstance(p, Parameter)) for pub in pubs]
        observables_vars = [ParameterTable.from_qiskit(pub.circuit, input_type=self._input_type, filter_function=lambda p: pub.circuit.has_var(p) and "obs" in p.name) for pub in pubs]
        # 2. PROGRAM CREATION: Pass the plans to the program builder.
        estimator_prog = estimator_program(
            self._backend, pubs, param_tables, observables_vars, execution_plans=self._execution_plans
        )
        self._program = estimator_prog
        compiler_options = self.metadata.get("compiler_options", None)

        # 3. EXECUTION: Start the QUA program on the OPX. It will wait for data.
        self._qm_job = self._backend.qm.execute(estimator_prog, compiler_options=compiler_options)
        self._job_id = self._qm_job.id

        # 4. DATA PUSHING: Loop through the planned tasks and push data to the running job.
        for i, pub in enumerate(self._pubs):
            plan = self._execution_plans[i]

            # This is the main loop that feeds the running QUA program
            for task_idx in range(plan.total_tasks):
                # Get the data for the current task from the plan
                param_idx = plan.task_param_indices[task_idx]
                param_values = plan.unique_params[param_idx]

                obs_indices_tuple = plan.task_obs_indices[task_idx]

                # Push parameter values if the circuit has them
                if param_tables[i] is not None and param_tables[i].input_type is not None and len(param_values) > 0:
                    param_dict = {param.name: value for param, value in zip(param_tables[i].parameters, param_values)}
                    param_tables[i].push_to_opx(param_dict, self.qm_job, self._backend.qm)

                if observables_vars[i] is not None and observables_vars[i].input_type is not None and len(obs_indices_tuple) > 0:
                    # Push observable indices for the measurement basis switch
                    obs_indices_dict = {f"obs_{i}": val for i, val in enumerate(obs_indices_tuple)}
                    observables_vars[i].push_to_opx(obs_indices_dict, self.qm_job, self._backend.qm)

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
