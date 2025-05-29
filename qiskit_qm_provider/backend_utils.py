from __future__ import annotations

from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, IfElseOp, WhileLoopOp, ForLoopOp, SwitchCaseOp
from qiskit.circuit.library import get_standard_gate_name_mapping

from quam.components import Qubit, QubitPair
from .additional_gates import CRGate, SYGate, SYdgGate
from quam_builder.architecture.superconducting.qpu.base_quam import BaseQuam
from oqc import OperationIdentifier


def validate_machine(machine) -> BaseQuam:
    if not hasattr(machine, "qubits") or not hasattr(machine, "qubit_pairs"):
        raise ValueError(
            "Invalid QuAM instance provided, should have qubits and qubit_pairs attributes"
        )
    if not all(isinstance(qubit, Qubit) for qubit in machine.qubits.values()):
        raise ValueError("All qubits should be of type Qubit")
    if not all(isinstance(qubit_pair, QubitPair) for qubit_pair in machine.qubit_pairs.values()):
        raise ValueError("All qubit pairs should be of type QubitPair")

    return machine


def validate_circuits(
    circuits: List[QuantumCircuit], should_reset: bool = True, check_for_params: bool = False
) -> List[QuantumCircuit]:
    """
    Validate the circuits to be compiled. The circuits should be a list of QuantumCircuits.
    :param circuits: List of QuantumCircuits to be validated.
    :param should_reset: If True, check if the circuit has a reset at the boundary.
    :param check_for_params: If True, check if the circuit has compile-time parameters.
    :return: Modified circuits with an added reset if needed.
    """
    if not all(isinstance(qc, QuantumCircuit) for qc in circuits):
        raise ValueError("Input should be a QuantumCircuit or a Qiskit Pulse Schedule")
    if check_for_params and not all(len(qc.parameters) == 0 for qc in circuits):
        raise ValueError("Input should not contain parameters")

    new_circuits = []
    for qc in circuits:
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) != 1:
                raise ValueError("Only one register per clbit is supported.")
        if not has_reset_at_boundary(qc) and should_reset:
            qc_reset = qc.copy_empty_like()
            qubits = [qc.qubits[i] for i in qc.layout.final_index_layout(filter_ancillas=True)]
            qc_reset.reset(qubits)
            new_circuits.append(qc.compose(qc_reset, inplace=False, front=True))
        else:
            new_circuits.append(qc)

    return new_circuits


def has_conflicting_calibrations(circuits: List[QuantumCircuit]) -> bool:
    """
    Check if the circuits have conflicting calibrations.
    :param circuits: List of QuantumCircuits to be checked.
    :return: True if there are conflicting calibrations, False otherwise.
    """
    custom_gates = []
    for qc in circuits:
        if hasattr(qc, "calibrations") and qc.calibrations:
            for gate_name, cal_info in qc.calibrations.items():
                for qubits, parameters in cal_info.keys():
                    op_id = OperationIdentifier(gate_name, len(parameters), qubits)
                    if op_id not in custom_gates:
                        custom_gates.append(op_id)
                    else:
                        return True
    return False


def look_for_standard_op(op: str):
    op = op.lower()
    mapping = {
        "cphase": "cz",
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
    return mapping.get(op, op)


def get_extended_gate_name_mapping():
    gate_map = get_standard_gate_name_mapping()

    gate_map["sy"] = SYGate()
    gate_map["cr"] = CRGate()
    gate_map["sydg"] = SYdgGate()

    return gate_map


def has_reset_at_boundary(circuit: QuantumCircuit) -> bool:
    """Check if the QuantumCircuit has a reset at the start or end."""
    instructions = circuit.data

    if not instructions:
        return False

    # Check first instruction
    first = instructions[0].operation.name == "reset"
    # Check last instruction
    last = instructions[-1].operation.name == "reset"

    return first or last


control_flow_name_mapping = {
    "if_else": IfElseOp,
    "while_loop": WhileLoopOp,
    "for_loop": ForLoopOp,
    "switch_case": SwitchCaseOp,
}
oq3_keyword_instructions = ("measure", "reset", "delay", "nop")
_QASM3_DUMP_LOOSE_BIT_PREFIX = "_bit"


def binary(val: int, num_bits: int = 0) -> str:
    """
    Convert an integer to a binary string with leading zeros.
    :param val: The integer value to convert.
    :param num_bits: The number of bits in the binary representation.
    :return: The binary string representation of the integer.
    """
    return bin(val)[2:].zfill(num_bits)


def add_basic_macros_to_machine(machine: BaseQuam):
    """
    Add macros to the machine.
    :param machine: The BaseQuam instance to which macros will be added.
    """
    from quam_libs.components.gate_macros import (
        ResetMacro,
        VirtualZMacro,
        MeasureMacro,
        CZMacro,
        DelayMacro,
    )
    from quam.components.macro import PulseMacro

    for qubit in machine.active_qubits:
        qubit.macros["x"] = PulseMacro(pulse="x180")
        qubit.macros["rz"] = VirtualZMacro()
        qubit.macros["sx"] = PulseMacro(pulse="x90")
        qubit.macros["measure"] = MeasureMacro(pulse="readout")
        qubit.macros["reset"] = ResetMacro(pi_pulse="x180", readout_pulse="readout")
        qubit.macros["delay"] = DelayMacro()

    for qubit_pair in machine.active_qubit_pairs:
        qubit_pair.macros["cz"] = CZMacro(
            flux_pulse_control=qubit_pair.qubit_control.get_pulse("flux_pulse").get_reference(),
            coupler_flux_pulse=qubit_pair.coupler.operations["cz"].get_reference(),
        )
