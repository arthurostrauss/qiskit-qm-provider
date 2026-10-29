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
from ..utils.calibrations.identifiers import has_conflicting_calibrations
from qiskit import QuantumCircuit
from qm import Program
from qm.qua import *
from ..parameter_table import ParameterTable
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .primitive_plan import EstimatorPlan, SamplerPlan


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


def _process_sampler_pub(plan: SamplerPlan, backend: QMBackend, **kwargs):
    """QUA macro: run one sampler PUB, looping over its bindings when it has parameters.

    Without parameters there is no table and no binding loop: the shot loop runs directly. Every
    binding is then the same circuit, so a bindings array of shape ``(k, 0)`` (``k`` bindings of
    no parameter) runs ``k`` times the shots, in binding order.
    """
    if plan.circuit_table is None:
        return _process_circuit(plan.pub.circuit, backend, plan.shots * plan.num_bindings, **kwargs)
    plan.declare_params()
    p = declare(int)
    with for_(p, 0, p < plan.num_bindings, p + 1):
        plan.load_params(p)
        outputs = _process_circuit(plan.pub.circuit, backend, plan.shots, plan.circuit_table, **kwargs)
    return outputs


def sampler_program(backend: QMBackend, plans: List[SamplerPlan], **kwargs) -> Program:
    """Return the QUA program for the given sampler plans."""
    # TODO: Handle OPNIC case where circuits share parameters (loading might not work)
    outputs_tables = []
    with program() as sampler_prog:
        backend.init_macro()
        for plan in plans:
            outputs_tables.append(_process_sampler_pub(plan, backend, **kwargs))
        with stream_processing():
            for i, outputs in enumerate(outputs_tables):
                for creg_name, field in outputs.table.items():
                    field.stream.buffer(plans[i].shots).save_all(f"{creg_name}_{i}")
    return sampler_prog


def _process_estimator_pub(plan: EstimatorPlan, backend: QMBackend, **kwargs):
    """QUA macro: run one estimator PUB, binding-major, one circuit run per measurement group.

    Per binding: load its parameters, then its packed observable row (a row shared by every
    binding is received once, before the loop), and unpack each group into the ``__obs_j``
    inputs before running the circuit. This matches :meth:`EstimatorPlan.push` order.
    """
    plan.declare_params()
    plan.declare_observables()
    tables = [t for t in (plan.circuit_table, plan.obs_table) if t is not None]
    if plan.circuit_table is None and plan.num_bindings == 1:
        # No parameters (the usual unparametrized PUB): no binding loop, only the groups.
        with plan.observable_groups(0):
            return _process_circuit(plan.pub.circuit, backend, plan.shots, tables, compute_state_int=False, **kwargs)
    # With parameters, or several bindings of no parameter (whose observables may differ).
    p = declare(int)
    with for_(p, 0, p < plan.num_bindings, p + 1):
        plan.load_params(p)
        with plan.observable_groups(p):
            outputs = _process_circuit(plan.pub.circuit, backend, plan.shots, tables, compute_state_int=False, **kwargs)
    return outputs


def estimator_program(backend: QMBackend, plans: List[EstimatorPlan], **kwargs) -> Program:
    """Return the QUA program for the given estimator plans."""
    outputs_tables = []
    with program() as estimator_prog:
        backend.init_macro()
        for plan in plans:
            outputs_tables.append(_process_estimator_pub(plan, backend, **kwargs))

        with stream_processing():
            for i, outputs in enumerate(outputs_tables):
                for creg_name, field in outputs.table.items():
                    bool_stream = field.stream.boolean_to_int()
                    # Array register → (shots, n_bits); scalar output (length 0) → (shots,).
                    if field.is_array:
                        bool_stream.buffer(plans[i].shots, field.length).save_all(f"{creg_name}_{i}")
                    else:
                        bool_stream.buffer(plans[i].shots).save_all(f"{creg_name}_{i}")

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
    outputs_list = []
    with program() as prog:
        backend.init_macro()

        for qc in circuits:
            outputs = _process_circuit(qc, backend, num_shots)
            outputs_list.append(outputs)

        with stream_processing():
            for i, outputs in enumerate(outputs_list):
                for creg_name, field in outputs.table.items():
                    field.stream.save_all(f"{creg_name}_{i}")

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


def compute_locator(chunk_layout: List[List[int]]) -> Dict[int, tuple]:
    """Invert a chunk layout into a ``global_index -> (chunk_idx, local_idx)`` dict."""
    return {
        g: (c, l)
        for c, chunk in enumerate(chunk_layout)
        for l, g in enumerate(chunk)
    }


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
    plans: List[SamplerPlan],
    **kwargs,
) -> tuple[List[Program], List[List[int]]]:
    """Plan the QUA program(s) for ``QMSamplerV2``.

    When the number of PUBs exceeds ``max_circuits``, the list is split into
    consecutive chunks, each compiled into its own QUA program.  Results are
    stitched back by the caller using the returned ``chunk_layout``.

    Within each program, the local PUB index (0-based within the chunk) is used
    for stream keys ``f"{creg_name}_{local_idx}"``.

    Plans with zero bindings are left out: they need no program (no programs and an empty
    layout when no plan has bindings).

    Args:
        backend: The QMBackend to use.
        plans: One :class:`~qiskit_qm_provider.job.primitive_plan.SamplerPlan` per pub.
        **kwargs: Forwarded to :func:`sampler_program`.

    Returns:
        Tuple of (list of QUA programs, chunk layout).
    """
    runnable = [g for g, plan in enumerate(plans) if plan.has_tasks]
    if not runnable:
        return [], []
    chunk_layout = [
        [runnable[i] for i in chunk]
        for chunk in compute_chunk_layout(len(runnable), max_circuits=backend.options.max_circuits)
    ]
    programs = [sampler_program(backend, [plans[g] for g in chunk], **kwargs) for chunk in chunk_layout]
    return programs, chunk_layout


def plan_estimator_programs(
    backend: QMBackend,
    plans: List[EstimatorPlan],
    **kwargs,
) -> tuple[List[Program], List[List[int]]]:
    """Plan the QUA program(s) for ``QMEstimatorV2``.

    Plans without tasks (only identity terms) are left out: they need no program. When the
    number of remaining plans exceeds ``backend.options.max_circuits``, they are split into
    consecutive chunks, each compiled into its own QUA program.  Results are stitched back by
    the caller using the returned ``chunk_layout`` (global plan indices; no programs and an
    empty layout when no plan has tasks).

    Within each program, the local plan index (0-based within the chunk) is
    used for stream keys ``f"__c_{local_idx}"``.

    Args:
        backend: The QMBackend to use.
        plans: One :class:`~qiskit_qm_provider.job.primitive_plan.EstimatorPlan` per pub.
        **kwargs: Forwarded to :func:`estimator_program`.

    Returns:
        Tuple of (list of QUA programs, chunk layout).
    """
    runnable = [g for g, plan in enumerate(plans) if plan.has_tasks]
    if not runnable:
        return [], []
    chunk_layout = [
        [runnable[i] for i in chunk]
        for chunk in compute_chunk_layout(len(runnable), max_circuits=backend.options.max_circuits)
    ]
    programs = [estimator_program(backend, [plans[g] for g in chunk], **kwargs) for chunk in chunk_layout]
    return programs, chunk_layout


def get_run_program(backend: QMBackend, num_shots, circuits: List[QuantumCircuit]) -> Program | List[Program]:
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
