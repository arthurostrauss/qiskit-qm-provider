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

"""Validate circuits and optionally prepend a boundary reset.

Author: Arthur Strauss
Date: 2026-10-29
"""

from __future__ import annotations

from typing import List

from qiskit.circuit import QuantumCircuit


def has_reset_at_boundary(circuit: QuantumCircuit) -> bool:
    """Check if each qubit in the QuantumCircuit has a reset at the start or end."""
    instructions = circuit.data
    qubits = circuit.qubits

    if not instructions:
        return True  # Empty circuit means all qubits are in reset state

    # Create per-qubit instruction lists
    qubit_instructions = {q: [] for q in qubits}
    for inst in instructions:
        for q in inst.qubits:
            qubit_instructions[q].append(inst)

    # Check each qubit's first and last operations
    for qubit_insts in list(qubit_instructions.values()):
        if not qubit_insts:
            continue  # No instructions means qubit remained in reset state

        # Check first operation on this qubit
        has_start_reset = qubit_insts[0].operation.name == "reset"

        # Check last operation on this qubit
        has_end_reset = qubit_insts[-1].operation.name == "reset"

        if not (has_start_reset or has_end_reset):
            return False

    return True


def logically_active_qubits(circuit):
    """
    Retrieve the qubits that are logically active in the circuit, meaning those that carry any operation other than a delay (related to scheduling passes).
    """

    active = set()

    for instruction in circuit.data:
        if instruction.operation.name == "delay":
            continue
        active.update(instruction.qubits)

    return sorted(active, key=lambda q: circuit.find_bit(q).index)


def validate_circuits(
    circuits: QuantumCircuit | List[QuantumCircuit],
    should_reset: bool = True,
    check_for_params: bool = False,
) -> List[QuantumCircuit]:
    """Validate circuits before compilation.

    Args:
        circuits: Single circuit or list of circuits to validate.
        should_reset: When ``True``, prepend a reset to circuits that lack one
            at the boundary.
        check_for_params: When ``True``, reject circuits with compile-time
            parameters.

    Returns:
        Validated circuits, with an automatic reset prepended when requested.

    Raises:
        ValueError: If inputs are invalid or classical-bit layout is unsupported.
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    if not all(isinstance(qc, QuantumCircuit) for qc in circuits):
        raise ValueError("Input should be a list of QuantumCircuits")
    if check_for_params and not all(len(qc.parameters) == 0 for qc in circuits):
        raise ValueError("Input should not contain parameters")

    new_circuits = []
    for qc in circuits:
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) > 1:
                raise ValueError("Only one register per clbit is supported.")
        if not has_reset_at_boundary(qc) and should_reset:
            qc_reset = qc.copy_empty_like(vars_mode="drop")
            active_qubits = logically_active_qubits(qc)
            qc_reset.reset(active_qubits)
            new_circuits.append(qc.compose(qc_reset, inplace=False))
        else:
            new_circuits.append(qc)

    return new_circuits
