from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, IfElseOp, WhileLoopOp, ForLoopOp, SwitchCaseOp
from qiskit.circuit.library import get_standard_gate_name_mapping

from quam.components import Qubit, QubitPair
from .additional_gates import CRGate, SYGate, SYdgGate
from quam_builder.architecture.superconducting.qpu.base_quam import BaseQuam


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
