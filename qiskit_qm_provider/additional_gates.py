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

"""Additional gates (SY, SYdg, CR) for Qiskit circuits targeting QM backends.

Author: Arthur Strauss
Date: 2026-02-08
"""

from qiskit.circuit import QuantumCircuit, Gate
import numpy as np
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping as gate_map

__all__ = [
    "SYGate",
    "SYdgGate",
    "CRGate",
]


class SYGate(Gate):
    def __init__(self, label=None):
        super().__init__("sy", 1, [], label=label)

    def _define(self):
        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        self._definition = qc

    def inverse(self, annotated: bool = False):
        qc = QuantumCircuit(1)
        qc.ry(-np.pi / 2, 0)
        return qc.to_gate()

    def power(self, exponent: float, annotated: bool = False):
        return gate_map()["ry"](np.pi / 2 * exponent)

    def __eq__(self, other):
        return isinstance(other, SYGate)

    def __array__(self, dtype=None, copy=None):
        return type(gate_map()["ry"])(np.pi / 2).__array__(dtype=dtype, copy=copy)


class SYdgGate(Gate):
    def __init__(self, label=None):
        super().__init__("sydg", 1, [], label=label)

    def _define(self):
        qc = QuantumCircuit(1)
        qc.ry(-np.pi / 2, 0)
        self._definition = qc

    def inverse(self, annotated: bool = False):
        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        return qc.to_gate()

    def power(self, exponent: float, annotated: bool = False):
        return gate_map()["ry"](-np.pi / 2 * exponent)

    def __eq__(self, other):
        return isinstance(other, SYdgGate)

    def __array__(self, dtype=None, copy=None):
        return type(gate_map()["ry"])(-np.pi / 2).__array__(dtype=dtype, copy=copy)


class CRGate(Gate):
    def __init__(self, label=None):
        super().__init__("cr", 2, [], label=label)

    def _define(self):
        from qiskit.circuit.library.standard_gates import RZXGate

        qc = QuantumCircuit(2, name=self.name)
        qc.append(RZXGate(np.pi / 2), [0, 1])
        self._definition = qc

    def inverse(self, annotated: bool = False):
        from qiskit.circuit.library.standard_gates import RZXGate

        # Equivalent to RZX(-pi/2)
        return RZXGate(-np.pi / 2)

    def power(self, exponent: float, annotated: bool = False):
        return gate_map()["rzx"](np.pi / 2 * exponent)

    def __eq__(self, other):
        return isinstance(other, CRGate)

    def __array__(self, dtype=None, copy=None):
        return type(gate_map()["rzx"])(np.pi / 2).__array__(dtype=dtype, copy=copy)


# Do monkey patching to QuantumCircuit to add the custom gates
# Add a method to QuantumCircuit to add the custom gates (qc.sy(q) == qc.append(SYGate(), [q]))
QuantumCircuit.sy = lambda self, q: self.append(SYGate(), [q])
QuantumCircuit.sydg = lambda self, q: self.append(SYdgGate(), [q])
QuantumCircuit.cr = lambda self, q1, q2: self.append(CRGate(), [q1, q2])
