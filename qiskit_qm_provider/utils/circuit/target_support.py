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

"""Check that a circuit uses only operations the backend Target supports.

Author: Arthur Strauss
Date: 2026-10-29
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes import GatesInBasis

from ...conditional_play import _raise_for_unregistered_conditional_plays

if TYPE_CHECKING:
    from qiskit.transpiler import Target


def _missing_names_in_dag(dag: DAGCircuit, target: Target) -> set[str]:
    supported = target.operation_names
    return {name for name in dag.count_ops(recurse=True) if name not in supported}


def missing_operation_names(qc: QuantumCircuit, target: Target) -> set[str]:
    """Return the names of operations in ``qc`` that ``target`` does not define at all.

    Counts operations with :meth:`DAGCircuit.count_ops` (``recurse=True``), so operations
    nested in control-flow blocks -- including every conditional play, which always sits
    inside a ``box`` -- are included; :meth:`QuantumCircuit.count_ops` would miss them.

    This is a name-only check: an operation whose name is in ``target`` but which is
    used on qubits or with parameters the target does not support is not reported (see
    :func:`ensure_native_operations` for the full check).

    Args:
        qc: QuantumCircuit to inspect.
        target: Target to compare against.

    Returns:
        The set of operation names in ``qc`` absent from ``target.operation_names``.
    """
    return _missing_names_in_dag(circuit_to_dag(qc, copy_operations=False), target)


def _first_unsupported_operation(qc: QuantumCircuit, target: Target, qubit_map: dict | None = None):
    """Return ``(operation, physical_qargs)`` for the first operation ``target`` rejects, else ``None``.

    Mirrors :class:`~qiskit.transpiler.passes.GatesInBasis`: parameters are only compared
    for non-parameterized, non-control-flow operations, and control-flow blocks are
    searched with their qubits mapped back to the physical qubits of the owning instruction.
    """
    if qubit_map is None:
        qubit_map = {qubit: index for index, qubit in enumerate(qc.qubits)}
    for instruction in qc.data:
        operation = instruction.operation
        qargs = tuple(qubit_map[qubit] for qubit in instruction.qubits)
        is_control_flow = isinstance(operation, ControlFlowOp)
        check_params = not is_control_flow and not operation.is_parameterized()
        if not target.instruction_supported(
            operation.name, qargs, parameters=list(operation.params) if check_params else None
        ):
            return operation, qargs
        if is_control_flow:
            for block in operation.blocks:
                found = _first_unsupported_operation(block, target, dict(zip(block.qubits, qargs)))
                if found is not None:
                    return found
    return None


def ensure_native_operations(qc: QuantumCircuit, target: Target) -> None:
    """Validate that every operation in ``qc`` is supported by ``target``.

    Uses Qiskit's :class:`~qiskit.transpiler.passes.GatesInBasis`, which checks each
    operation's name, physical qargs and parameters against ``target``, recursing into
    control-flow blocks (``GatesInBasis`` itself always accepts ``barrier`` and ``store``;
    QM backend Targets also define both, see ``variadic_builtin_instructions``). ``qc`` must
    already be physical (see :func:`~qiskit_qm_provider.utils.circuit.physical_layout.ensure_circuit_physical`)
    so that its qargs are physical qubit indices. ``qc`` is only inspected, never modified.

    When the check fails, the error is built from the cheapest diagnosis that explains
    it: unregistered conditional plays first (see
    :func:`~qiskit_qm_provider.conditional_play._raise_for_unregistered_conditional_plays`),
    then operation names ``target`` does not define at all (:func:`missing_operation_names`),
    and only if every name is defined, the first operation used on unsupported qubits or
    with unsupported parameters.

    Args:
        qc: QuantumCircuit to check.
        target: Target the circuit is compiled against.

    Raises:
        ValueError: If any operation is not supported by ``target``.
    """
    dag = circuit_to_dag(qc, copy_operations=False)
    gates_in_basis = GatesInBasis(target=target)
    # Run the analysis pass directly: going through ``__call__`` would build a
    # PassManager and convert the DAG back into a throwaway circuit.
    gates_in_basis.run(dag)
    if gates_in_basis.property_set["all_gates_in_basis"]:
        return

    missing_names = _missing_names_in_dag(dag, target)
    _raise_for_unregistered_conditional_plays(qc, missing_names)
    if missing_names:
        raise ValueError(
            f"Circuit {qc.name!r} uses operation(s) {sorted(missing_names)} that the given Target "
            "does not define. Transpile the circuit against that target before compiling it."
        )

    # Every name is defined, so the mismatch is in qubits or parameters: find it.
    unsupported = _first_unsupported_operation(qc, target)
    detail = (
        f": {unsupported[0].name!r} on qubit(s) {unsupported[1]} with params {list(unsupported[0].params)}"
        if unsupported is not None
        else ""
    )
    raise ValueError(
        f"Circuit {qc.name!r} uses an operation on qubits or with parameters the given Target "
        f"does not support{detail}. Transpile the circuit against that target before compiling it."
    )
