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
from ..parameter_table import (
    InputType,
    ParameterTable,
    ParameterPool,
    Parameter as QuaParameter,
)
from .iqcc_job_mixin import IQCCJobMixin
from .qua_programs import plan_estimator_programs
from .qm_primitive_job import QMPrimitiveJob
from ..primitives.qm_estimator import QMEstimatorOptions
from dataclasses import dataclass, field
from collections import defaultdict
import os
import inspect
import tempfile

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
    param_group_keys: List[tuple] = field(default_factory=list)

    @classmethod
    def from_pub(cls, pub: EstimatorPub, options: QMEstimatorOptions):
        from ..backend.backend_utils import (
            logically_active_qubits,
            get_non_trivial_observables,
        )

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
        param_group_keys = []
        for param_index, pauli_strings in param_obs_map.items():
            param_group_keys.append(param_index)
            meas_paulis = PauliList(sorted(pauli_strings))
            meas_paulis_active = get_non_trivial_observables(
                meas_paulis, [circuit.find_bit(q).index for q in active_qubits]
            )
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
                    metadata.append(
                        {
                            "meas_paulis": paulis,
                            "param_index": param_index,
                            "orig_paulis": obs,
                        }
                    )
            else:
                for basis in meas_paulis:
                    _, indices = _measurement_circuit(circuit.num_qubits, basis)
                    obs = PauliList(basis)
                    paulis = PauliList.from_symplectic(
                        obs.z[:, indices],
                        obs.x[:, indices],
                        obs.phase,
                    )
                    metadata.append(
                        {
                            "meas_paulis": paulis,
                            "param_index": param_index,
                            "orig_paulis": obs,
                        }
                    )
        observables_var = ParameterTable.from_qiskit(
            pub.circuit,
            input_type=options.input_type,
            filter_function=lambda p: pub.circuit.has_var(p) and "obs" in p.name,
            name=f"observables_var_{pub.circuit.name}",
        )
        param_table = ParameterTable.from_qiskit(
            pub.circuit,
            input_type=options.input_type,
            filter_function=lambda p: isinstance(p, Parameter),
            name=f"param_table_{pub.circuit.name}",
        )
        return cls(
            pub,
            metadata,
            param_indices,
            obs_indices_list,
            observables_var,
            param_table,
            active_qubits,
            param_group_keys,
        )

    @property
    def total_tasks(self) -> int:
        return sum(len(obs_indices_list) for obs_indices_list in self.obs_indices)

    @property
    def uses_compile_time_param_bindings(self) -> bool:
        """``True`` when circuit parameters are preloaded in QUA (``input_type is None``)."""
        return (
            self.param_table is not None
            and bool(self.param_table.parameters)
            and self.param_table.input_type is None
        )

    @property
    def stream_buffer_count(self) -> int:
        """Number of ``(shots, n_bits)`` buffers saved to the ``__c`` stream.

        With a parameter sweep, each binding runs only its own observable tasks,
        so buffers accumulate as ``sum(len(obs_indices[p]))``.
        """
        if self.param_table is None or not self.param_table.parameters:
            return self.total_tasks
        return sum(len(group) for group in self.obs_indices)

    def stream_indices_for_metadata(self) -> List[int]:
        """Map each metadata task to its row in the flattened ``__c`` stream."""
        obs_row_starts: List[int] = []
        offset = 0
        for group in self.obs_indices:
            obs_row_starts.append(offset)
            offset += len(group)

        has_param_loop = self.param_table is not None and bool(self.param_table.parameters)
        meta_count_per_param: dict[tuple, int] = defaultdict(int)
        stream_offsets_per_binding: List[int] = [0]
        for group in self.obs_indices:
            stream_offsets_per_binding.append(stream_offsets_per_binding[-1] + len(group))

        indices: List[int] = []
        for meta in self.metadata:
            param_index = meta["param_index"]
            local_meta = meta_count_per_param[param_index]
            meta_count_per_param[param_index] += 1
            if has_param_loop:
                p = _param_binding_index(self.param_indices, param_index)
                indices.append(stream_offsets_per_binding[p] + local_meta)
            else:
                group_idx = self.param_group_keys.index(param_index)
                indices.append(obs_row_starts[group_idx] + local_meta)
        return indices

    @property
    def shots(self) -> int:
        return int(np.ceil(1 / self.pub.precision**2))

    @property
    def num_qubits(self) -> int:
        return len(self.active_qubits)

    @property
    def active_qubit_indices(self) -> List[int]:
        return [self.pub.circuit.find_bit(q).index for q in self.active_qubits]


def _param_binding_index(param_indices: np.ndarray, param_index: tuple) -> int:
    for idx, candidate in np.ndenumerate(param_indices):
        if candidate == param_index:
            return int(np.ravel_multi_index(idx, param_indices.shape))
    raise ValueError(f"param_index {param_index!r} not found in execution plan")


def counts_from_estimator_stream(plan: _ExecutionPlan, raw) -> List[Counts]:
    """Build per-metadata :class:`~qiskit.result.Counts` from the ``__c`` stream."""
    data = np.asarray(raw)
    shots = plan.shots
    num_qubits = plan.num_qubits
    unit = shots * num_qubits
    if data.size % unit != 0:
        raise ValueError(
            f"Estimator stream size {data.size} is not a multiple of "
            f"shots×num_qubits ({shots}×{num_qubits}={unit})."
        )
    stream_count = data.size // unit
    expected = plan.stream_buffer_count
    if stream_count != expected:
        raise ValueError(
            f"Estimator stream has {stream_count} buffers but expected {expected} "
            f"(compile-time params={plan.uses_compile_time_param_bindings})."
        )

    bitstrings = data.reshape(stream_count, shots, num_qubits)
    bitstrings_bool = np.asarray(bitstrings, dtype=bool)
    bit_array = BitArray.from_bool_array(bitstrings_bool)
    return [
        Counts(bit_array.get_counts(loc=(stream_idx,)))
        for stream_idx in plan.stream_indices_for_metadata()
    ]


def observables_to_indices(
    observables: List[SparsePauliOp | Pauli | str] | SparsePauliOp | PauliList | Pauli | str,
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
    """Job handle for :class:`~qiskit_qm_provider.primitives.QMEstimatorV2` execution.

    Builds a QUA estimator program from pubs and returns expectation values via
    :meth:`result`. See :attr:`programs` for the compiled QUA source and
    :attr:`result_handles` for raw QM stream access after submit.
    """

    def result(self) -> ResultT:
        """Build and return primitive estimator results from QM streaming data.

        Returns:
            :class:`~qiskit.primitives.PrimitiveResult` with per-pub expectation
            values and standard errors.
        """
        if self._qm_jobs is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._result_function(self._qm_jobs)

    @property
    def runtime_pubs(self) -> "List[_ExecutionPlan]":
        """Compiled execution plans for this estimator job.

        One :class:`_ExecutionPlan` per input PUB.  Each plan holds the grouped
        observable metadata, parameter table, and obs-index tables that drive the
        QUA switch statement and parameter streaming.  Inspect to understand how
        observables were grouped or what will be streamed cycle-by-cycle.
        """
        return self._execution_plans

    def __init__(
        self,
        backend: QMBackend,
        pubs: List[EstimatorPub],
        input_type: Optional[InputType],
        switch_obs_circuit: QuantumCircuit,
        **kwargs,
    ):
        """Create an estimator job.

        Args:
            backend: Backend that compiled the circuits.
            pubs: Coerced estimator pubs to execute.
            input_type: How circuit parameters are streamed to the OPX.
            switch_obs_circuit: Pre-transpiled circuit used to rotate observables.
            **kwargs: Estimator options forwarded to execution planning.
        """
        super().__init__(backend, pubs, input_type, **kwargs)
        ParameterPool.reset()
        self._execution_plans: List[_ExecutionPlan] = [
            _ExecutionPlan.from_pub(pub, options=QMEstimatorOptions(input_type=input_type, **kwargs)) for pub in pubs
        ]
        self._switch_obs_circuit: QuantumCircuit = switch_obs_circuit
        self._obs_length_vars = QuaParameter(
            name="obs_length_var", value=0, qua_type=int, input_type=input_type
        )
        programs, self._chunk_layout = plan_estimator_programs(
            backend,
            self._execution_plans,
            obs_length_var=self._obs_length_vars,
        )
        self._programs = programs
        # Locator: global plan index -> (chunk_program_index, local_plan_index)
        self._locator = {
            g: (c, l)
            for c, chunk in enumerate(self._chunk_layout)
            for l, g in enumerate(chunk)
        }

    def _push_plan_data(self, qm_job, plan: "_ExecutionPlan") -> None:
        """Stream parameters and observable indices for one execution plan to OPX."""
        param_table = plan.param_table
        observables_var = plan.observables_var

        if param_table is not None and param_table.input_type is not None:
            for p, param_value in enumerate(plan.pub.parameter_values.ravel().as_array()):
                param_dict = {
                    param.name: value
                    for param, value in zip(param_table.parameters, param_value)
                }
                param_table.push_to_opx(param_dict, qm_job, self._backend.qm)
                if observables_var.input_type is not None:
                    self._obs_length_vars.push_to_opx(
                        len(plan.obs_indices[p]), qm_job, self._backend.qm
                    )
                    for obs_value in plan.obs_indices[p]:
                        obs_dict = {f"obs_{j}": val for j, val in enumerate(obs_value)}
                        observables_var.push_to_opx(obs_dict, qm_job, self._backend.qm)
        elif observables_var.input_type is not None:
            self._obs_length_vars.push_to_opx(
                len(plan.obs_indices[0]), qm_job, self._backend.qm
            )
            for obs_value in plan.obs_indices[0]:
                obs_dict = {f"obs_{j}": val for j, val in enumerate(obs_value)}
                observables_var.push_to_opx(obs_dict, qm_job, self._backend.qm)

    def submit(self):
        """Submit the job to the backend.

        When execution plans were split into multiple QUA programs (chunked
        execution), each program is queued sequentially on QOP.  Results from
        all chunks are transparently stitched back in :meth:`_result_function`.
        """
        if self._qm_jobs is not None:
            raise RuntimeError("Job has already been submitted.")
        compiler_options = self.metadata.get("compiler_options", None)
        simulate = self.metadata.get("simulate", None)

        programs = self._programs

        if simulate is not None and isinstance(self._backend.qmm, QuantumMachinesManager):
            job = self._backend.qmm.simulate(
                self._backend.qm_config,
                programs[0],
                simulate=simulate,
                compiler_options=compiler_options,
            )
            self._qm_jobs = [job]
            self._job_id = job.id
            for global_idx in self._chunk_layout[0]:
                self._push_plan_data(job, self._execution_plans[global_idx])
        elif len(programs) == 1:
            job = self._backend.qm.execute(programs[0], compiler_options=compiler_options)
            self._qm_jobs = [job]
            self._job_id = job.id
            for global_idx in self._chunk_layout[0]:
                self._push_plan_data(job, self._execution_plans[global_idx])
        else:
            self._qm_jobs = []
            for prog, chunk in zip(programs, self._chunk_layout):
                job = self._backend.qm.queue.add(prog, compiler_options=compiler_options)
                self._qm_jobs.append(job)
                for global_idx in chunk:
                    self._push_plan_data(job, self._execution_plans[global_idx])
            self._job_id = ",".join(j.id for j in self._qm_jobs)

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
                    expval_map[param_index, orig_pauli.to_label()] = (
                        float(expval),
                        float(variance),
                    )
            else:
                # Length mismatch - this shouldn't happen normally, but handle it
                # Use first expval for all orig_paulis (grouped measurement case)
                for orig_pauli in orig_paulis_list:
                    expval_map[param_index, orig_pauli.to_label()] = (
                        float(expvals[0]),
                        float(variances[0]),
                    )

        return expval_map

    def _postprocess_pub(
        self,
        pub: EstimatorPub,
        expval_map: Dict,
        param_indices: np.ndarray,
        observables: np.ndarray,
        shots: int,
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

    def _result_function(self, qm_jobs: List[RunningQmJob]) -> PrimitiveResult[PubResult]:
        results_handles = [job.result_handles for job in qm_jobs]
        for handle in results_handles:
            handle.wait_for_all_values()

        pub_results = []
        for i, plan in enumerate(self._execution_plans):
            chunk_idx, local_idx = self._locator[i]
            handle = results_handles[chunk_idx]
            raw = handle.get(f"__c_{local_idx}").fetch_all()
            counts_list = counts_from_estimator_stream(plan, raw)

            # Compute expectation value map
            expval_map = self._calc_expval_map(counts_list, plan.metadata)

            # Postprocess to get PubResult
            # Convert observables to numpy array for broadcasting
            # plan.pub.observables is an ObservablesArray, we need to broadcast it
            _, bc_obs = np.broadcast_arrays(plan.param_indices, plan.pub.observables)

            pub_result = self._postprocess_pub(plan.pub, expval_map, plan.param_indices, bc_obs, plan.shots)
            pub_results.append(pub_result)

        return PrimitiveResult(pub_results, metadata={"version": 2})

    def _run(self):
        pass

    def _parse_result(self):
        pass


class IQCCEstimatorJob(IQCCJobMixin, QMEstimatorJob):
    """IQCC Primitive Job class for executing QUA programs from PUBs."""

    def submit(self):
        """Submit the job to the backend.

        When execution plans were split into multiple QUA programs (chunked
        execution), each program is submitted as a separate IQCC cloud job with
        its own sync hook written to a system temp file.  Results are stitched
        back transparently by the inherited :meth:`_result_function` using the
        locator built at construction time.
        """
        from .post_hook_estimator import generate_sync_hook_estimator

        if self._qm_jobs is not None:
            raise RuntimeError("IQCC QM job has already been submitted")

        programs = self._programs
        timeout = self.metadata.get("run_options", {}).get("timeout", None)
        jobs = []

        for prog, chunk in zip(programs, self._chunk_layout):
            chunk_plans = [self._execution_plans[g] for g in chunk]

            sync_hook_path = None
            if any(p.param_table is not None and p.param_table.input_type is not None for p in chunk_plans):
                sync_hook_code = generate_sync_hook_estimator(chunk_plans, obs_length_var=self._obs_length_vars)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(sync_hook_code)
                    sync_hook_path = f.name
                options = {"sync_hook": sync_hook_path}
            else:
                options = {}

            if timeout is not None:
                options["timeout"] = timeout

            try:
                jobs.append(self._backend.qm.execute(prog, options=options))  # type: ignore
            finally:
                if sync_hook_path is not None:
                    os.unlink(sync_hook_path)

        self._qm_jobs = jobs
        self._job_id = ",".join(getattr(j, "id", "") for j in jobs)
