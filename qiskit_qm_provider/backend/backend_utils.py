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

"""Backend utilities: circuit-to-QUA translation, calibration handling, measurement outcomes.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

import warnings
from typing import List, TYPE_CHECKING, Dict, Literal, Type

from qiskit import QuantumCircuit
from qiskit.circuit.controlflow import (
    ControlFlowOp,
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    SwitchCaseOp,
)
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Pauli, PauliList

from quam.components import Qubit, QubitPair
from quam.core import QuamRoot
from quam.utils.qua_types import QuaVariableInt
from ..additional_gates import CRGate, SYGate, SYdgGate
from qm.qua import declare, assign, Cast, declare_stream

if TYPE_CHECKING:
    from qm_qasm import CompilationResult
    from .qm_backend import QMBackend
try:
    from qiskit.circuit.controlflow import get_control_flow_name_mapping

    control_flow_name_mapping = get_control_flow_name_mapping()
except ImportError:
    warnings.warn(
        "get_control_flow_name_mapping is not available in this version of Qiskit, skipping it from control flow mapping."
    )
    control_flow_name_mapping: Dict[str, Type[ControlFlowOp]] = {
        "if_else": IfElseOp,
        "while_loop": WhileLoopOp,
        "for_loop": ForLoopOp,
        "switch_case": SwitchCaseOp,
    }

qasm3_keyword_instructions = (
    "measure",
    "reset",
    "delay",
    "nop",
    "box",
    "for_loop",
    "while_loop",
    "if_else",
    "switch_case",
)
_QASM3_DUMP_LOOSE_BIT_PREFIX = "_bit"


def validate_machine(machine) -> QuamRoot:
    """
    Validate the QuAM instance by checking if it has qubits and qubit_pairs attributes and if they are of type Qubit and QubitPair respectively.
    :param machine: The QuAM instance to be validated.
    :return: The QuAM instance if it is valid, otherwise raises a ValueError.
    """
    if not hasattr(machine, "qubits") or not hasattr(machine, "qubit_pairs"):
        raise ValueError(
            "Invalid QuAM instance provided, should have qubits and qubit_pairs attributes"
        )
    if not all(isinstance(qubit, Qubit) for qubit in machine.qubits.values()):
        raise ValueError("All qubits should be of type Qubit")
    if not all(
        isinstance(qubit_pair, QubitPair) for qubit_pair in machine.qubit_pairs.values()
    ):
        raise ValueError("All qubit pairs should be of type QubitPair")

    return machine


def validate_circuits(
    circuits: QuantumCircuit | List[QuantumCircuit],
    should_reset: bool = True,
    check_for_params: bool = False,
) -> List[QuantumCircuit]:
    """
    Validate the circuits to be compiled. The circuits should be a list of QuantumCircuits.
    :param circuits: List of QuantumCircuits to be validated.
    :param should_reset: If True, check if the circuit has a reset at the boundary.
    :param check_for_params: If True, check if the circuit has compile-time parameters.
    :return: Modified circuits with an added reset if needed.
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
            if len(qc.find_bit(clbit).registers) != 1:
                raise ValueError("Only one register per clbit is supported.")
        if not has_reset_at_boundary(qc) and should_reset:
            qc_reset = qc.copy_empty_like(vars_mode="drop")
            active_qubits = logically_active_qubits(qc)
            qc_reset.reset(active_qubits)
            new_circuits.append(qc.compose(qc_reset, inplace=False))
        else:
            new_circuits.append(qc)

    return new_circuits


def has_conflicting_calibrations(circuits: List[QuantumCircuit]) -> bool:
    """
    Check if the circuits have conflicting calibrations.
    :param circuits: List of QuantumCircuits to be checked.
    :return: True if there are conflicting calibrations, False otherwise.
    """
    from qm_qasm import OperationIdentifier

    custom_gates = set()
    for qc in circuits:
        if hasattr(qc, "calibrations") and qc.calibrations:
            for gate_name, cal_info in qc.calibrations.items():
                for qubits, parameters in cal_info.keys():
                    op_id = OperationIdentifier(gate_name, len(parameters), qubits)
                    if op_id not in custom_gates:
                        custom_gates.add(op_id)
                    else:
                        return True
    return False


def look_for_standard_op(op: str):
    op = op.lower()
    mapping = {
        "cphase": "cp",
        "cnot": "cx",
        "x/2": "sx",
        "x90": "sx",
        "x180": "x",
        "y180": "y",
        "-x/2": "sxdg",
        "-x90": "sxdg",
        "y90": "sy",
        "y/2": "sy",
        "-y/2": "sydg",
        "-y90": "sydg",
        "hadamard": "h",
        "identity": "id",
        "wait": "delay",
        "readout": "measure",
        "meas": "measure",
        "zz": "rzz",
        "yy": "ryy",
        "xx": "rxx",
    }
    for key in mapping.keys():
        if key in op:
            return mapping[key]
    return mapping.get(op, op)


def get_extended_gate_name_mapping():
    gate_map = get_standard_gate_name_mapping()

    gate_map["sy"] = SYGate()
    gate_map["cr"] = CRGate()
    gate_map["sydg"] = SYdgGate()

    return gate_map


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


def binary(val: int, num_bits: int = 0) -> str:
    """
    Convert an integer to a binary string with leading zeros.
    :param val: The integer value to convert.
    :param num_bits: The number of bits in the binary representation.
    :return: The binary string representation of the integer.
    """
    return bin(val)[2:].zfill(num_bits)


def add_basic_macros(
    backend: QuamRoot | QMBackend,
    reset_type: Literal["active", "thermalize"] = "thermalize",
):
    """
    Add macros to the machine.
    :param machine: The BaseQuam instance to which macros will be added.
    :param reset_type: The type of reset to use. Can be 'active' or 'thermalize'.
    """
    from iqcc_calibration_tools.quam_config.components.gate_macros import (
        ResetMacro,
        VirtualZMacro,
        MeasureMacro,
        DelayMacro,
        IdMacro,
    )
    from quam.components.macro import PulseMacro
    from quam_builder.architecture.superconducting.custom_gates.flux_tunable_transmon_pair.two_qubit_gates import (
        CZGate,
    )
    from .qm_backend import QMBackend

    if not isinstance(backend, (QuamRoot, QMBackend)):
        raise ValueError("Backend should be a QuamRoot or QMBackend instance")
    machine = backend.machine if isinstance(backend, QMBackend) else backend

    for qubit in machine.active_qubits:
        x180_pulse = qubit.get_pulse("x180").get_reference()
        readout_pulse = qubit.get_pulse("readout").get_reference()
        x90_pulse = qubit.get_pulse("x90").get_reference()
        y90_pulse = qubit.get_pulse("y90").get_reference()
        my90_pulse = qubit.get_pulse("-y90").get_reference()
        qubit.macros["x"] = PulseMacro(pulse=x180_pulse)
        qubit.macros["rz"] = VirtualZMacro()
        qubit.macros["sx"] = PulseMacro(pulse=x90_pulse)
        qubit.macros["sy"] = PulseMacro(pulse=y90_pulse)
        qubit.macros["sydg"] = PulseMacro(pulse=my90_pulse)
        qubit.macros["measure"] = MeasureMacro(pulse=readout_pulse)
        qubit.macros["reset"] = ResetMacro(
            reset_type=reset_type, pi_pulse=x180_pulse, readout_pulse=readout_pulse
        )
        qubit.macros["delay"] = DelayMacro()
        qubit.macros["id"] = IdMacro()

    for qubit_pair in machine.active_qubit_pairs:
        try:
            qubit_pair.macros["cz"] = None
            qubit_pair.macros["cz"] = CZGate(
                flux_pulse_control=qubit_pair.qubit_control.z.operations[
                    "const"
                ].get_reference(),
            )
        except ValueError as e:
            warnings.warn(
                f"Could not add default two qubit gates. Add it manually if necessary. Error: {e}"
            )


def get_measurement_outcomes(
    qc: QuantumCircuit, result: CompilationResult, compute_state_int: bool = True
) -> dict[str, dict[str, QuaVariableInt]]:
    """
    Get the measurement outcomes resulting from the execution of the QuantumCircuit.
    This is returned as a dictionary of the form {creg_name: {"value": [outcome_values], "state_int": state_int, "size": size}}, where state_int is a QUA variable that contains the integer representation of each ClassicalRegister belonging to the QuantumCircuit and size is the size of the ClassicalRegister.
    Note that this follows the Qiskit convention of using the least significant bit (LSB) of the integer to represent the state of the qubit with index 0.
    We also support the case where the QuantumCircuit contains loose bits (bits that are not associated with a ClassicalRegister).
    In this case, an extra ClassicalRegister is added to the QuantumCircuit with the name _bit, containing the measurement outcomes of the loose bits with the same convention.
    :param compute_state_int: If True, compute the state integer for each ClassicalRegister.
    """
    clbits_dict = {
        creg.name: {
            "value": result.result_program[creg.name],
            "stream": declare_stream(),
            "size": creg.size,
        }
        for creg in qc.cregs
    }
    num_solo_bits = len(
        [bit for bit in qc.clbits if len(qc.find_bit(bit).registers) == 0]
    )
    if num_solo_bits > 0:
        clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX] = {
            "value": [
                result.result_program[f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{i}"]
                for i in range(num_solo_bits)
            ],
            "stream": declare_stream(),
            "size": num_solo_bits,
        }
    if compute_state_int:
        for creg_dict in clbits_dict.values():
            c_reg_res = creg_dict["value"]
            creg_dict["state_int"] = declare(int)
            assign(
                creg_dict["state_int"],
                sum(
                    (
                        ((1 << i) * Cast.to_int(c_reg_res[i]))
                        for i in range(1, creg_dict["size"])
                    ),
                    start=Cast.to_int(c_reg_res[0]),
                ),
            )
    return clbits_dict


def logically_active_qubits(circuit):
    """
    Retrieve the qubits that are logically active in the circuit, meaning those that carry any operation other than a delay (related to scheduling passes).
    """

    active = set()

    for instr, qargs, _ in circuit.data:
        if instr.name == "delay":
            continue
        for q in qargs:
            active.add(q)

    return sorted(active, key=lambda q: circuit.find_bit(q).index)


def get_non_trivial_observables(
    observables: PauliList, active_qubit_indices: List[int]
) -> PauliList:
    """
    Get the non-trivial observables from the observables.
    :param observables: The observables to get the non-trivial observables from.
    :param active_qubit_indices: The indices of the active qubits.
    :return: The non-trivial observables.
    """
    new_observables = []
    for observable in observables:
        label = observable.to_label()
        new_pauli_label = ""
        for j in range(observable.num_qubits):
            if j in active_qubit_indices:
                new_pauli_label += label[-j - 1]
        new_observables.append(Pauli(new_pauli_label))

    return PauliList(new_observables)
