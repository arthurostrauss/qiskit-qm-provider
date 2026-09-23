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

"""Additional gates (SY, SYdg, CR, fSim, GPI, GPI2) for Qiskit circuits targeting QM backends.

Author: Arthur Strauss
Date: 2026-02-08
"""

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping as gate_map,
)
import numpy as np


__all__ = [
    "SYGate",
    "SYdgGate",
    "CRGate",
    "FSimGate",
    "GPIGate",
    "GPI2Gate",
]


class SYGate(Gate):
    """
    Quantum Machines sqrt-Y gate: a fixed π/2 rotation about the Y axis.

    Unitary in the computational basis {|0⟩, |1⟩}:

        ┌                              ┐
        │  1/√2     -1/√2              │
        │  1/√2      1/√2              │
        └                              ┘

    Equivalent to RY(π/2).
    """

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
    """
    Quantum Machines sqrt-Y-dagger gate: a fixed -π/2 rotation about the Y axis.

    Unitary in the computational basis {|0⟩, |1⟩}:

        ┌                              ┐
        │   1/√2      1/√2             │
        │  -1/√2      1/√2             │
        └                              ┘

    Equivalent to RY(-π/2). Inverse of SY.
    """

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
    """
    Quantum Machines CR gate: a fixed 2-qubit XX interaction.

    Unitary in the computational basis {|00⟩, |01⟩, |10⟩, |11⟩}:

        ┌                                          ┐
        │  1/√2       0          0        -i/√2    │
        │    0     1/√2      -i/√2         0       │
        │    0    -i/√2       1/√2         0       │
        │ -i/√2       0          0        1/√2    │
        └                                          ┘

    Equivalent to RZX(π/2).
    """

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


class FSimGate(Gate):
    """
    Google fermionic Simulation (fSim) gate: a 2-qubit gate parametrized by (θ, φ).

    Unitary in the computational basis {|00⟩, |01⟩, |10⟩, |11⟩}:

        ┌                              ┐
        │  1       0          0      0 │
        │  0    cos(θ)   -i·sin(θ)   0 │
        │  0   -i·sin(θ)  cos(θ)    0 │
        │  0       0          0    e^{-iφ} │
        └                              ┘

    Special cases:
        θ=π/4, φ=0   → √iSWAP
        θ=π/2, φ=0   → iSWAP
        θ=0,   φ=π   → CZ  (up to local phases)
    """

    def __init__(self, theta: ParameterValueType, phi: ParameterValueType, label=None):
        super().__init__("fsim", num_qubits=2, params=[theta, phi], label=label)

    def _define(self):
        """Decompose fSim(theta, phi) into standard Qiskit two-qubit rotations."""
        theta, phi = self.params
        qc = QuantumCircuit(2, name=self.name)
        qc.rxx(theta, 0, 1)
        qc.ryy(theta, 0, 1)
        qc.cp(-phi, 0, 1)
        self._definition = qc

    def __array__(self, dtype=None):
        theta, phi = float(self.params[0]), float(self.params[1])
        cos = np.cos(theta)
        sin = np.sin(theta)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, -1j * sin, 0.0],
                [0.0, -1j * sin, cos, 0.0],
                [0.0, 0.0, 0.0, np.exp(-1j * phi)],
            ],
            dtype=complex if dtype is None else dtype,
        )

    def inverse(self, annotated: bool = False) -> "FSimGate":
        """
        Returns the inverse of the FSimGate.

        Args:
            None

        Returns:
            A FSimGate instance with the parameters inverted (θ -> -θ, φ -> -φ)
        """
        return FSimGate(-self.params[0], -self.params[1])

    def power(self, exponent: float, annotated: bool = False) -> "FSimGate":
        """

        Returns the power of the FSimGate.

        Args:
            exponent: The exponent to raise the parameters to

        Returns:
            A FSimGate instance with the parameters raised to the power of the exponent (θ -> θ^exponent, φ -> φ^exponent)
        """
        return FSimGate(exponent * self.params[0], exponent * self.params[1])

class GPIGate(Gate):
    r"""Qibo's ``GPI`` gate: :math:`\begin{pmatrix} 0 & e^{-i\phi} \\
    e^{i\phi} & 0 \end{pmatrix}`.

    An IonQ-native gate with no Qiskit standard-library equivalent.
    """

    def __init__(self, phi: ParameterValueType, label: str | None = None):
        super().__init__("gpi", 1, [phi], label=label)

    def _define(self) -> None:
        # Qibo's own qasm_label body: u3(pi, phi - pi/2, pi/2 - phi)
        phi = self.params[0]
        qc = QuantumCircuit(1, name=self.name)
        qc.u(np.pi, phi - np.pi / 2, np.pi / 2 - phi, 0)
        self.definition = qc

    def __array__(self, dtype=complex, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        phi = complex(self.params[0])
        return np.array([[0, np.exp(-1j * phi)], [np.exp(1j * phi), 0]], dtype=dtype)


class GPI2Gate(Gate):
    r"""Qibo's ``GPI2`` gate: :math:`\frac{1}{\sqrt2}\begin{pmatrix}
    1 & -i e^{-i\phi} \\ -i e^{i\phi} & 1 \end{pmatrix}`.

    An IonQ-native gate with no Qiskit standard-library equivalent.
    """

    def __init__(self, phi: ParameterValueType, label: str | None = None):
        super().__init__("gpi2", 1, [phi], label=label)

    def _define(self) -> None:
        # Qibo's own qasm_label body: u3(pi/2, phi - pi/2, pi/2 - phi)
        phi = self.params[0]
        qc = QuantumCircuit(1, name=self.name)
        qc.u(np.pi / 2, phi - np.pi / 2, np.pi / 2 - phi, 0)
        self.definition = qc

    def __array__(self, dtype=complex, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        phi = complex(self.params[0])
        return np.array(
            [[1, -1j * np.exp(-1j * phi)], [-1j * np.exp(1j * phi), 1]],
            dtype=dtype,
        ) / np.sqrt(2)



# Do monkey patching to QuantumCircuit to add the custom gates
# Add a method to QuantumCircuit to add the custom gates (qc.sy(q) == qc.append(SYGate(), [q]))
QuantumCircuit.sy = lambda self, q, label=None: self.append(SYGate(label), [q])
QuantumCircuit.sydg = lambda self, q, label=None: self.append(SYdgGate(label), [q])
QuantumCircuit.cr = lambda self, q1, q2, label=None: self.append(CRGate(label), [q1, q2])
QuantumCircuit.fsim = lambda self, theta, phi, q1, q2, label=None: self.append(FSimGate(theta, phi, label), [q1, q2])
QuantumCircuit.gpi = lambda self, phi, q, label=None: self.append(GPIGate(phi, label), [q])
QuantumCircuit.gpi2 = lambda self, phi, q, label=None: self.append(GPI2Gate(phi, label), [q])
