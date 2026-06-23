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

"""QUA program builders for sampler and estimator from Qiskit circuits and PUBs.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from ..backend import QMBackend
from ..backend.backend_utils import has_conflicting_calibrations
from qiskit import QuantumCircuit
from qm import Program
from qm.qua import *
from qiskit.primitives.containers.sampler_pub import SamplerPub
from ..parameter_table import ParameterTable, QUA2DArray
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .qm_estimator_job import _ExecutionPlan


def _estimator_process_param_table(plan: _ExecutionPlan):
    if plan.param_table is not None:
        return [plan.param_table, plan.observables_var]
    return plan.observables_var


def _obs_indices_flat(plan: _ExecutionPlan) -> np.ndarray:
    return np.vstack([np.array(group, dtype=np.int32) for group in plan.obs_indices])


def _declare_obs_indices_all(plan: _ExecutionPlan, name: str) -> QUA2DArray:
    obs_indices_all = QUA2DArray(name, _obs_indices_flat(plan), qua_type=int)
    obs_indices_all.declare()
    return obs_indices_all


def _declare_compile_time_obs_arrays(plan: _ExecutionPlan, name_prefix: str):
    """Length, offset, and index tables for a compile-time parameter sweep."""
    obs_lengths = np.array([len(group) for group in plan.obs_indices], dtype=np.int32)
    obs_offsets = np.concatenate([[0], np.cumsum(obs_lengths[:-1])]).astype(np.int32)
    obs_lengths_qua = QUA2DArray(
        f"obs_lengths_{name_prefix}",
        obs_lengths.reshape(-1, 1),
        qua_type=int,
    )
    obs_offsets_qua = QUA2DArray(
        f"obs_offsets_{name_prefix}",
        obs_offsets.reshape(-1, 1),
        qua_type=int,
    )
    obs_indices_all = QUA2DArray(
        f"obs_indices_{name_prefix}",
        _obs_indices_flat(plan),
        qua_type=int,
    )
    obs_lengths_qua.declare()
    obs_offsets_qua.declare()
    obs_indices_all.declare()
    return obs_lengths_qua, obs_offsets_qua, obs_indices_all


def _process_circuit(
    qc: QuantumCircuit,
    backend: QMBackend,
    num_shots: int,
    param_table: Optional[ParameterTable | List[ParameterTable]] = None,
    compute_state_int: bool = True,
    **kwargs,
):
    shot_var = declare(int)

    with for_(shot_var, 0, shot_var < num_shots, shot_var + 1):
        comp = backend.quantum_circuit_to_qua(qc, param_table)
        outputs = comp.outputs
        # Save each register / loose bit to its compiler-owned stream.
        for field in outputs.parameters:
            if compute_state_int:
                save(field.state_int, field.stream)
            elif field.is_array:
                loop_var = declare(int)
                with for_(loop_var, 0, loop_var < field.length, loop_var + 1):
                    save(field.var[loop_var], field.stream)
            else:
                # Scalar output (e.g. a loose clbit) — save the single value directly.
                save(field.var, field.stream)

    return outputs


def _process_sampler_pub(
    pub: SamplerPub,
    backend: QMBackend,
    param_table: Optional[ParameterTable] = None,
    **kwargs,
):
    """
    Process the given PUB with the given backend.
    If the parameter table is not provided, the circuit is processed without parameters.
    If the parameter table is provided, the circuit is processed with parameters.
    This is a QUA macro that is used to process the circuit.
    """
    if param_table is None:
        outputs = _process_circuit(
            pub.circuit,
            backend,
            pub.shots,
            **kwargs,
        )
    else:
        param_table.declare()
        p = declare(int)

        with for_(p, 0, p < pub.parameter_values.ravel().size, p + 1):
            if param_table.input_type is None:
                # Declare the parameters at compile time
                param_values_qua = QUA2DArray("param_values", pub.parameter_values.ravel().as_array())
                param_values_qua.declare()
                param_table.assign_parameters(
                    {param.name: param_values_qua[p][i] for i, param in enumerate(param_table.parameters)}
                )
            else:
                param_table.rcv()
            outputs = _process_circuit(
                pub.circuit,
                backend,
                pub.shots,
                param_table,
                **kwargs,
            )
    return outputs


def sampler_program(
    backend: QMBackend,
    pubs: List[SamplerPub],
    param_tables: List[ParameterTable],
    **kwargs,
) -> Program:
    """Return the QUA program for the given PUBs."""
    circuits = [pub.circuit for pub in pubs]
    num_circuits = len(circuits)
    # TODO: Handle OPNIC case where circuits share parameters (loading might not work)
    outputs_tables = []
    with program() as sampler_prog:
        backend.init_macro()

        for i in range(num_circuits):
            outputs = _process_sampler_pub(
                pubs[i],
                backend,
                param_tables[i],
                **kwargs,
            )
            outputs_tables.append(outputs)
        with stream_processing():
            for i, outputs in enumerate(outputs_tables):
                for creg_name, field in outputs.table.items():
                    field.stream.buffer(pubs[i].shots).save_all(f"{creg_name}_{i}")
    return sampler_prog


def _process_observables_with_circuit(
    plan: _ExecutionPlan,
    backend: QMBackend,
    **kwargs,
):
    """
    QUA macro to process observables and execute the circuit.
    Handles both cases where observables_var.input_type is None (compile-time)
    or not None (runtime input).

    Args:
        plan: The ExecutionPlan containing the circuit
        backend: The QMBackend to use
        observables_var: ParameterTable for observables
        param_table: Optional additional ParameterTable to pass to _process_circuit
        **kwargs: Additional arguments for _process_circuit
    """
    obs_length_var = kwargs.pop("obs_length_var", None)
    param_binding_var = kwargs.pop("param_binding_var", None)
    obs_lengths_qua = kwargs.pop("obs_lengths_qua", None)
    obs_offsets_qua = kwargs.pop("obs_offsets_qua", None)
    obs_indices_all = kwargs.pop("obs_indices_all", None)
    obs_idx = declare(int)
    num_qubits = len(plan.active_qubit_indices)
    plan.observables_var.declare()
    if obs_length_var is not None:
        obs_length_var.declare()

    process_param_table = _estimator_process_param_table(plan)
    circuit_kwargs = {
        "param_table": process_param_table,
        "compute_state_int": False,
    }

    if plan.observables_var.input_type is None:
        per_binding = param_binding_var is not None and obs_lengths_qua is not None
        if obs_indices_all is None:
            obs_indices_all = _declare_obs_indices_all(plan, "obs_indices_all")
        obs_loop_count = (
            obs_lengths_qua[param_binding_var, 0] if per_binding else obs_indices_all.n_rows
        )

        with for_(obs_idx, 0, obs_idx < obs_loop_count, obs_idx + 1):
            obs_row = (
                obs_offsets_qua[param_binding_var, 0] + obs_idx if per_binding else obs_idx
            )
            plan.observables_var.assign_parameters(
                {f"obs_{i}": obs_indices_all[obs_row, i] for i in range(num_qubits)}
            )
            outputs = _process_circuit(plan.pub.circuit, backend, plan.shots, **circuit_kwargs)
    else:
        if obs_length_var is not None:
            obs_length_var.rcv()
        with for_(obs_idx, 0, obs_idx < obs_length_var.var, obs_idx + 1):
            plan.observables_var.rcv()
            outputs = _process_circuit(plan.pub.circuit, backend, plan.shots, **circuit_kwargs)

    return outputs


def _process_estimator_pub(
    plan: _ExecutionPlan,
    backend: QMBackend,
    **kwargs,
):
    """
    Process the given EstimatorPub with the given backend.
    If the parameter table is not provided, the circuit is processed without parameters.
    If the parameter table is provided, the circuit is processed with parameters.
    This is a QUA macro that is used to process the circuit.
    """
    if plan.param_table is None:
        # Process observables only (no additional param_table)
        outputs = _process_observables_with_circuit(
            plan,
            backend,
            **kwargs,
        )
    else:
        p = declare(int)
        plan.param_table.declare()
        param_values_qua = None
        obs_lengths_qua = obs_offsets_qua = obs_indices_all = None
        if plan.param_table.input_type is None:
            param_values_qua = QUA2DArray(
                f"param_values_{plan.param_table.name}",
                plan.pub.parameter_values.ravel().as_array(),
            )
            param_values_qua.declare()
            if plan.observables_var.input_type is None:
                obs_lengths_qua, obs_offsets_qua, obs_indices_all = _declare_compile_time_obs_arrays(
                    plan, plan.param_table.name
                )

        with for_(p, 0, p < plan.pub.parameter_values.ravel().size, p + 1):
            if plan.param_table.input_type is None:
                plan.param_table.assign_parameters(
                    {param.name: param_values_qua[p, i] for i, param in enumerate(plan.param_table.parameters)}
                )
            else:
                plan.param_table.rcv()

            outputs = _process_observables_with_circuit(
                plan,
                backend,
                param_binding_var=p if obs_lengths_qua is not None else None,
                obs_lengths_qua=obs_lengths_qua,
                obs_offsets_qua=obs_offsets_qua,
                obs_indices_all=obs_indices_all,
                **kwargs,
            )
    return outputs


def estimator_program(backend: QMBackend, execution_plans: List[_ExecutionPlan], **kwargs) -> Program:
    """
    Return the QUA program for the estimator primitive based on pre-computed plans.
    """
    # The execution plans are generated in the 'submit' method and passed here.
    outputs_tables = []
    with program() as estimator_prog:
        backend.init_macro()

        # Loop over each PUB/circuit
        for plan in execution_plans:
            outputs = _process_estimator_pub(
                plan,
                backend,
                **kwargs,
            )
            outputs_tables.append(outputs)

        with stream_processing():
            for i, outputs in enumerate(outputs_tables):
                for creg_name, field in outputs.table.items():
                    bool_stream = field.stream.boolean_to_int()
                    # Array register → (shots, n_bits); scalar output (length 0) → (shots,).
                    if field.is_array:
                        bool_stream.buffer(execution_plans[i].shots, field.length).save_all(f"{creg_name}_{i}")
                    else:
                        bool_stream.buffer(execution_plans[i].shots).save_all(f"{creg_name}_{i}")

    return estimator_prog


def _build_single_program(
    backend: QMBackend, num_shots, circuits: List[QuantumCircuit]
) -> Program:
    """Build one QUA program executing ``circuits`` sequentially.

    Each circuit's classical registers are saved to streams named
    ``f"{creg_name}_{local_index}"`` where ``local_index`` is the position of
    the circuit *within this program* (0-based). The caller is responsible for
    mapping those local indices back to global circuit indices when stitching
    results (see :func:`plan_run_programs`).
    """
    clbits_dicts = []
    with program() as prog:
        backend.init_macro()

        for qc in circuits:
            clbits_dict = _process_circuit(
                qc,
                backend,
                num_shots,
            )
            clbits_dicts.append(clbits_dict)

        with stream_processing():
            for i, clbits_dict in enumerate(clbits_dicts):
                for creg_name, creg_dict in clbits_dict.items():
                    creg_dict["stream"].save_all(f"{creg_name}_{i}")

    return prog


def compute_chunk_layout(
    num_circuits: int,
    max_circuits: int,
    conflicting_calibrations: bool = False,
) -> List[List[int]]:
    """Compute the chunk layout (lists of global circuit indices per program).

    Pure index arithmetic, independent of QUA program building, so it can be
    unit-tested in isolation. ``chunk_layout[c]`` lists the global circuit
    indices packed into program ``c``.

    Args:
        num_circuits: Total number of circuits (or PUBs/plans).
        max_circuits: Maximum circuits per program. Must be >= 1.
        conflicting_calibrations: When ``True``, each circuit gets its own
            program (takes priority over ``max_circuits``).

    Rules, in priority order:
        1. ``conflicting_calibrations`` -> one circuit per program.
        2. ``num_circuits > max_circuits`` -> consecutive groups of ``max_circuits``.
        3. Otherwise -> a single program holding all circuits.
    """
    if not isinstance(max_circuits, int) or max_circuits < 1:
        raise ValueError(f"max_circuits must be a positive integer (>= 1), got {max_circuits!r}")
    if conflicting_calibrations:
        return [[i] for i in range(num_circuits)]
    if num_circuits > max_circuits:
        return [
            list(range(start, min(start + max_circuits, num_circuits)))
            for start in range(0, num_circuits, max_circuits)
        ]
    return [list(range(num_circuits))]


def plan_run_programs(
    backend: QMBackend,
    num_shots,
    circuits: List[QuantumCircuit],
) -> tuple[List[Program], List[List[int]]]:
    """Plan the QUA program(s) for ``backend.run``.

    Splits ``circuits`` into one or more QUA programs and returns them together
    with a ``chunk_layout`` describing which global circuit indices live in each
    program. ``chunk_layout[c]`` is the list of global indices packed into
    ``programs[c]`` (in order), so ``programs[c]`` saves the result of
    ``chunk_layout[c][l]`` under stream key ``f"{creg}_{l}"``.

    Splitting rules, in priority order:
        1. Conflicting calibrations -> one circuit per program.
        2. ``backend.options.max_circuits`` set and ``len(circuits) > max_circuits`` ->
           consecutive groups of ``max_circuits``.
        3. Otherwise -> a single program holding all circuits.

    Args:
        backend: The QMBackend to use.
        num_shots: Number of shots to execute per circuit.
        circuits: List of QuantumCircuits to execute.

    Returns:
        Tuple of (list of QUA programs, chunk layout).
    """
    chunk_layout = compute_chunk_layout(
        len(circuits),
        max_circuits=backend.options.max_circuits,
        conflicting_calibrations=has_conflicting_calibrations(circuits),
    )

    programs = [
        _build_single_program(backend, num_shots, [circuits[g] for g in chunk])
        for chunk in chunk_layout
    ]
    return programs, chunk_layout


def plan_sampler_programs(
    backend: QMBackend,
    pubs: List[SamplerPub],
    param_tables: List[ParameterTable],
    **kwargs,
) -> tuple[List[Program], List[List[int]]]:
    """Plan the QUA program(s) for ``QMSamplerV2``.

    When the number of PUBs exceeds ``max_circuits``, the list is split into
    consecutive chunks, each compiled into its own QUA program.  Results are
    stitched back by the caller using the returned ``chunk_layout``.

    Within each program, the local PUB index (0-based within the chunk) is used
    for stream keys ``f"{creg_name}_{local_idx}"``.

    Args:
        backend: The QMBackend to use.
        pubs: List of SamplerPub objects.
        param_tables: Matching list of ParameterTable objects (one per pub).
        **kwargs: Forwarded to :func:`sampler_program`.

    Returns:
        Tuple of (list of QUA programs, chunk layout).
    """
    chunk_layout = compute_chunk_layout(len(pubs), max_circuits=backend.options.max_circuits)
    programs = [
        sampler_program(
            backend,
            [pubs[g] for g in chunk],
            [param_tables[g] for g in chunk],
            **kwargs,
        )
        for chunk in chunk_layout
    ]
    return programs, chunk_layout


def plan_estimator_programs(
    backend: QMBackend,
    execution_plans: List["_ExecutionPlan"],
    **kwargs,
) -> tuple[List[Program], List[List[int]]]:
    """Plan the QUA program(s) for ``QMEstimatorV2``.

    When the number of execution plans (one per EstimatorPub) exceeds
    ``backend.options.max_circuits``, the list is split into consecutive chunks,
    each compiled into its own QUA program.  Results are stitched back by the
    caller using the returned ``chunk_layout``.

    Within each program, the local plan index (0-based within the chunk) is
    used for stream keys ``f"__c_{local_idx}"``.

    Args:
        backend: The QMBackend to use.
        execution_plans: Pre-computed ``_ExecutionPlan`` objects (one per pub).
        **kwargs: Forwarded to :func:`estimator_program`.

    Returns:
        Tuple of (list of QUA programs, chunk layout).
    """
    chunk_layout = compute_chunk_layout(len(execution_plans), max_circuits=backend.options.max_circuits)
    programs = [
        estimator_program(backend, [execution_plans[g] for g in chunk], **kwargs)
        for chunk in chunk_layout
    ]
    return programs, chunk_layout


def get_run_program(
    backend: QMBackend, num_shots, circuits: List[QuantumCircuit]
) -> Program | List[Program]:
    """
    Build the QUA program(s) used by :meth:`~qiskit_qm_provider.backend.qm_backend.QMBackend.run`.

    Backward-compatible wrapper around :func:`plan_run_programs`.  Splitting
    behaviour is controlled by ``backend.options.max_circuits``.

    Args:
        backend: The QMBackend to use.
        num_shots: Number of shots to execute per circuit.
        circuits: List of QuantumCircuits to execute.

    Returns:
        A single :class:`~qm.Program`, or a list of programs when splitting occurs.
    """
    programs, _ = plan_run_programs(backend, num_shots, circuits)
    return programs[0] if len(programs) == 1 else programs
