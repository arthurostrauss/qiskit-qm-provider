from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import get_standard_gate_name_mapping

from quam.components import BasicQuam as QuAM, Qubit, QubitPair


def validate_machine(machine) -> QuAM:
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
        "y90": "sy",
        "hadamard": "h",
        "identity": "id",
        "wait": "delay",
        "readout": "measure",
    }
    return mapping.get(op, op)


def get_extended_gate_name_mapping():
    gate_map = get_standard_gate_name_mapping()

    class SYGate(Gate):
        def __init__(self, label=None):
            super().__init__("sy", 1, [], label=label)

        def _define(self):
            qc = QuantumCircuit(1)
            qc.ry(np.pi / 2, 0)
            self.definition = qc

        def inverse(self, annotated: bool = False):
            qc = QuantumCircuit(1)
            qc.ry(-np.pi / 2, 0)
            return qc.to_gate()

        def __eq__(self, other):
            return isinstance(other, SYGate)

        def __array__(self, dtype=None, copy=None):
            if copy is False:
                raise ValueError("unable to avoid copy while creating an array as requested")
            return gate_map["ry"](np.pi / 2).__array__(dtype=dtype, copy=copy)

    class CRGate(Gate):
        def __init__(self, label=None):
            super().__init__("cr", 2, [], label=label)

        def _define(self):
            from qiskit.circuit.library.standard_gates import RZXGate

            qc = QuantumCircuit(2, name=self.name)
            qc.append(RZXGate(np.pi / 2), [0, 1])
            self.definition = qc

        def inverse(self, annotated: bool = False):
            from qiskit.circuit.library.standard_gates import RZXGate

            # Equivalent to RZX(-pi/2)
            return RZXGate(-np.pi / 2)

        def __eq__(self, other):
            return isinstance(other, CRGate)

        def __array__(self, dtype=None, copy=None):
            if copy is False:
                raise ValueError("unable to avoid copy while creating an array as requested")
            return gate_map["rzx"](np.pi / 2).__array__(dtype=dtype, copy=copy)

    gate_map["sy"] = SYGate()
    gate_map["cr"] = CRGate()

    return gate_map
