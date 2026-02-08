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

"""QM estimator job: runs estimator PUBs as QUA programs and returns expectation values.

Author: Arthur Strauss
Date: 2026-02-08
"""

import numpy as np
from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit, Qubit
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.backend_estimator_v2 import (
    _measurement_circuit,
    _pauli_expval_with_variance,
)
from qiskit.primitives.base.base_primitive_job import ResultT
from qiskit.primitives.containers import DataBin, BitArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers import PubResult
from qiskit.result import Counts
from qm import SimulationConfig, CompilerOptionArguments, QuantumMachinesManager
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Union, List, Dict, Tuple, TYPE_CHECKING
from ..backend import QMBackend
from ..parameter_table import InputType, ParameterTable, ParameterPool, Parameter as QuaParameter
from .qua_programs import estimator_program
from .qm_primitive_job import QMPrimitiveJob
from ..primitives.qm_estimator import QMEstimatorOptions
from dataclasses import dataclass, field
from collections import defaultdict
import os
import inspect
if TYPE_CHECKING:
    from iqcc_cloud_client.qmm_cloud import CloudJob

@dataclass
class _ExecutionPlan:
    """Holds the pre-computed execution plan for a single EstimatorPub."""

    pub: EstimatorPub
    metadata: List[Dict]
    param_indices: np.ndarray
    obs_indices: List[List[Tuple[int, ...]]]
    observables_var: ParameterTable
    param_table: Optional[ParameterTable]
    active_qubits: List[Qubit]

    @classmethod
    def from_pub(cls, pub: EstimatorPub, options: QMEstimatorOptions):
        from ..backend.backend_utils import logically_active_qubits, get_non_trivial_observables
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)

        active_qubits = logically_active_qubits(circuit)
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
            meas_paulis_active = get_non_trivial_observables(meas_paulis, [circuit.find_bit(q).index for q in active_qubits])
            obs_indices_list.append(observables_to_indices(meas_paulis_active))
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
        observables_var = ParameterTable.from_qiskit(pub.circuit, input_type=options.input_type, filter_function=lambda p: pub.circuit.has_var(p) and "obs" in p.name, name=f"observables_var_{pub.circuit.name}")
        param_table = ParameterTable.from_qiskit(pub.circuit, input_type=options.input_type, filter_function=lambda p: isinstance(p, Parameter), name=f"param_table_{pub.circuit.name}")
        return cls(pub, metadata, param_indices, obs_indices_list, observables_var, param_table, active_qubits)

    @property
    def total_tasks(self) -> int:
        return sum(len(obs_indices_list) for obs_indices_list in self.obs_indices)
    
    @property
    def shots(self) -> int:
        return int(np.ceil(1/self.pub.precision**2))

    @property
    def num_qubits(self) -> int:
        return len(self.active_qubits)

    @property
    def active_qubit_indices(self) -> List[int]:
        return [self.pub.circuit.find_bit(q).index for q in self.active_qubits]

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
    @property
    def result_handles(self):
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._qm_job.result_handles
    def result(self) -> ResultT:
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._result_function(self._qm_job)

    def __init__(self, backend: QMBackend, pubs: List[EstimatorPub], input_type: Optional[InputType], switch_obs_circuit: QuantumCircuit, **kwargs):
        super().__init__(backend, pubs, input_type, **kwargs)
        ParameterPool.reset()
        self._execution_plans: List[_ExecutionPlan] = [_ExecutionPlan.from_pub(pub, options=QMEstimatorOptions(input_type=input_type, **kwargs)) for pub in pubs]
        self._switch_obs_circuit: QuantumCircuit = switch_obs_circuit
        self._obs_length_vars = QuaParameter(name="obs_length_var", value=0, qua_type=int, input_type=input_type)
        self._program = estimator_program(backend, self._execution_plans, obs_length_var=self._obs_length_vars)

        
    def submit(self):
        """Submit the job to the backend after creating an efficient execution plan."""
        if self._qm_job is not None:
            raise RuntimeError("Job has already been submitted.")
    
        compiler_options = self.metadata.get("compiler_options", None)
        simulate = self.metadata.get("simulate", None)
        # 3. EXECUTION: Start the QUA program on the OPX. It will wait for data.
        if simulate is not None and isinstance(self._backend.qmm, QuantumMachinesManager):
            self._qm_job = self._backend.qmm.simulate(self._backend.qm_config,estimator_prog, simulate=simulate, compiler_options=compiler_options)
            self._job_id = self._qm_job.id
        else:
            self._qm_job = self._backend.qm.execute(estimator_prog, compiler_options=compiler_options)
            self._job_id = self._qm_job.id

        
        # 4. DATA PUSHING: Loop through the planned tasks and push data to the running job.
        for i, plan in enumerate(self._execution_plans):
            plan = self._execution_plans[i]
            param_table = plan.param_table
            observables_var = plan.observables_var
            
            # Push parameter values if the circuit has them
            if param_table is not None and param_table.input_type is not None:
                for p, param_value in enumerate(plan.pub.parameter_values.ravel().as_array()):
                    param_dict = {param.name: value for param, value in zip(param_table.parameters, param_value)}
                    param_table.push_to_opx(param_dict, self.qm_job, self._backend.qm)
                    if observables_var.input_type is not None:
                        self._obs_length_vars.push_to_opx(len(plan.obs_indices[p]), self.qm_job, self._backend.qm)
                        for obs_value in plan.obs_indices[p]:
                            obs_dict = {f"obs_{i}": val for i, val in enumerate(obs_value)}
                            observables_var.push_to_opx(obs_dict, self.qm_job, self._backend.qm)
                
            # Push observable indices
            elif observables_var.input_type is not None:
                self._obs_length_vars.push_to_opx(len(plan.obs_indices[0]), self.qm_job, self._backend.qm)
                for obs_value in plan.obs_indices[0]:
                    obs_dict = {f"obs_{i}": val for i, val in enumerate(obs_value)}
                    observables_var.push_to_opx(obs_dict, self.qm_job, self._backend.qm)

    def _calc_expval_map(
        self,
        counts: List[Counts],
        metadata: List[Dict],
    ) -> Dict[Tuple[Tuple[int, ...], str], Tuple[float, float]]:
        """Computes the map of expectation values from Counts objects.

        Args:
            counts: List of Counts objects, one per task
            metadata: List of metadata dicts, one per task

        Returns:
            The map of expectation values takes a pair of an index of the bindings array and
            a pauli string as a key and returns the expectation value and variance of the pauli string.
        """
        expval_map: Dict[Tuple[Tuple[int, ...], str], Tuple[float, float]] = {}
        
        for task_idx, meta in enumerate(metadata):
            orig_paulis = meta["orig_paulis"]
            meas_paulis = meta["meas_paulis"]
            param_index = meta["param_index"]
            
            # Get counts for this task
            count = counts[task_idx]
            
            # Compute expectation values using the same method as backend_estimator_v2
            expvals, variances = _pauli_expval_with_variance(count, meas_paulis)
            
            # Map back to original Paulis
            # orig_paulis can be a SparsePauliOp (with coefficients) or PauliList
            # meas_paulis and orig_paulis should be in the same order
            if isinstance(orig_paulis, SparsePauliOp):
                # orig_paulis is a SparsePauliOp, iterate over its paulis
                orig_paulis_list = orig_paulis.paulis
            elif isinstance(orig_paulis, PauliList):
                # orig_paulis is a PauliList
                orig_paulis_list = orig_paulis
            else:
                # Fallback: treat as single Pauli
                if isinstance(orig_paulis, Pauli):
                    orig_paulis_list = PauliList([orig_paulis])
                else:
                    # Try to convert to PauliList
                    orig_paulis_list = PauliList([Pauli(str(orig_paulis))])
            
            # Map expectation values: meas_paulis and orig_paulis_list should be in same order
            if len(meas_paulis) == len(orig_paulis_list):
                # Direct 1-to-1 mapping
                for orig_pauli, expval, variance in zip(orig_paulis_list, expvals, variances):
                    expval_map[param_index, orig_pauli.to_label()] = (float(expval), float(variance))
            else:
                # Length mismatch - this shouldn't happen normally, but handle it
                # Use first expval for all orig_paulis (grouped measurement case)
                for orig_pauli in orig_paulis_list:
                    expval_map[param_index, orig_pauli.to_label()] = (float(expvals[0]), float(variances[0]))
        
        return expval_map

    def _postprocess_pub(
        self, 
        pub: EstimatorPub, 
        expval_map: Dict, 
        param_indices: np.ndarray,
        observables: np.ndarray,
        shots: int
    ) -> PubResult:
        """Computes expectation values (evs) and standard errors (stds).

        The values are stored in arrays broadcast to the shape of the pub.

        Args:
            pub: The pub to postprocess.
            expval_map: The map of expectation values.
            param_indices: The parameter indices array.
            observables: The observables array broadcast to pub shape.
            shots: The number of shots.

        Returns:
            The pub result.
        """
        bc_param_ind = param_indices
        bc_obs = observables
        evs = np.zeros_like(bc_param_ind, dtype=float)
        variances = np.zeros_like(bc_param_ind, dtype=float)
        
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            for pauli, coeff in bc_obs[index].items():
                key = (param_index, pauli)
                if key in expval_map:
                    expval, variance = expval_map[key]
                    evs[index] += expval * coeff
                    variances[index] += np.abs(coeff) * variance**0.5
        
        stds = variances / np.sqrt(shots)
        data_bin = DataBin(evs=evs, stds=stds, shape=evs.shape)
        return PubResult(
            data_bin,
            metadata={
                "target_precision": pub.precision,
                "shots": shots,
                "circuit_metadata": pub.circuit.metadata,
            },
        )

    def _result_function(self, qm_job: Union[RunningQmJob, List[QmPendingJob]]) -> PrimitiveResult[PubResult]:
        is_job_list = isinstance(qm_job, list)
        if is_job_list:
            results_handle = [job.result_handles for job in qm_job]
            for handle in results_handle:
                handle.wait_for_all_values()
        else:
            results_handle = qm_job.result_handles
            results_handle.wait_for_all_values()

        pub_results = []
        for i, plan in enumerate(self._execution_plans):
            if is_job_list:
                data = results_handle[i].get(f"__c_{i}").fetch_all()["value"]
            else:
                data = results_handle.get(f"__c_{i}").fetch_all()["value"]
            
            num_qubits = len(plan.active_qubits)
            shots = plan.shots
            total_tasks = plan.total_tasks
            
            bitstrings = np.array(data).reshape(total_tasks, shots, num_qubits)
            
            # Convert bitstrings to BitArray and then to Counts objects
            # Convert to boolean if needed (QUA might return integers 0/1)
            bitstrings_bool = np.asarray(bitstrings, dtype=bool)
            # BitArray.from_bool_array expects: (pub_shape..., shots, bits)
            # Shape: (total_tasks, shots, num_qubits) -> pub_shape=(total_tasks,), shots=shots, bits=num_qubits
            bit_array = BitArray.from_bool_array(bitstrings_bool)
            
            # Get Counts object for each task
            counts_list = []
            for task_idx in range(total_tasks):
                # Get counts for this specific task using loc parameter
                task_counts_dict = bit_array.get_counts(loc=(task_idx,))
                counts_list.append(Counts(task_counts_dict))
            
            # Compute expectation value map
            expval_map = self._calc_expval_map(counts_list, plan.metadata)
            
            # Postprocess to get PubResult
            # Convert observables to numpy array for broadcasting
            # plan.pub.observables is an ObservablesArray, we need to broadcast it
            _, bc_obs = np.broadcast_arrays(plan.param_indices, plan.pub.observables)
            
            pub_result = self._postprocess_pub(
                plan.pub,
                expval_map,
                plan.param_indices,
                bc_obs,
                shots
            )
            pub_results.append(pub_result)
        
        return PrimitiveResult(pub_results, metadata={"version": 2})


    def _run(self):
        pass

    def _parse_result(self):
        pass


class IQCCEstimatorJob(QMEstimatorJob):
    """IQCC Primitive Job class for executing QUA programs from PUBs."""

    def submit(self):
        """Submit the job to the backend."""
        from .post_hook_estimator import generate_sync_hook_estimator
        
        estimator_prog = self._program
        if self._qm_job is not None:
            raise RuntimeError("IQCC QM job has already been submitted")
        
        if any(plan.param_table is not None and plan.param_table.input_type is not None for plan in self._execution_plans):
            sync_hook_code = generate_sync_hook_estimator(self._execution_plans, obs_length_var=self._obs_length_vars)
        else:
            sync_hook_code = None

        # Determine the calling context to get the script file path
        caller_frame = inspect.stack()[-1]
        main_script_path = caller_frame.filename
        main_script_dir = os.path.dirname(os.path.abspath(main_script_path))
        sync_hook_path = os.path.join(main_script_dir, "sync_hook_estimator.py")
        if sync_hook_code is not None:
            with open(sync_hook_path, "w") as f:
                f.write(sync_hook_code)
            options = {"sync_hook": sync_hook_path}
        else:
            options = {}
        timeout = self.metadata["run_options"].get('timeout', None)
        if timeout is not None:
            options["timeout"] = timeout
        # # For IQCC, execute returns CloudJob instead of RunningQmJob
        self._qm_job = self._backend.qm.execute(  # type: ignore
            estimator_prog, options=options
        )

    def _result_function(self, qm_job: "CloudJob") -> PrimitiveResult[PubResult]:  # type: ignore[override]
        """Get the result from the IQCC QM job."""
        results_handle = qm_job.result_handles
        results_handle.wait_for_all_values()

        pub_results = []
        for i, plan in enumerate(self._execution_plans):
            # For IQCC, result_handle.get() returns data directly, similar to sampler
            data = np.array(results_handle.get(f"__c_{i}").fetch_all())
            num_qubits = len(plan.active_qubits)
            shots = plan.shots
            total_tasks = plan.total_tasks
            
            bitstrings = data.reshape((total_tasks, shots, num_qubits))
            
            # Convert bitstrings to BitArray and then to Counts objects
            # Convert to boolean if needed (QUA might return integers 0/1)
            bitstrings_bool = np.asarray(bitstrings, dtype=bool)
            # BitArray.from_bool_array expects: (pub_shape..., shots, bits)
            # Shape: (total_tasks, shots, num_qubits) -> pub_shape=(total_tasks,), shots=shots, bits=num_qubits
            bit_array = BitArray.from_bool_array(bitstrings_bool)
            
            # Get Counts object for each task
            counts_list = []
            for task_idx in range(total_tasks):
                # Get counts for this specific task using loc parameter
                task_counts_dict = bit_array.get_counts(loc=(task_idx,))
                counts_list.append(Counts(task_counts_dict))
            
            # Compute expectation value map
            expval_map = self._calc_expval_map(counts_list, plan.metadata)
            
            # Postprocess to get PubResult
            _, bc_obs = np.broadcast_arrays(plan.param_indices, plan.pub.observables)
            
            pub_result = self._postprocess_pub(
                plan.pub,
                expval_map,
                plan.param_indices,
                bc_obs,
                shots
            )
            pub_results.append(pub_result)
        
        return PrimitiveResult(pub_results, metadata={"version": 2})
