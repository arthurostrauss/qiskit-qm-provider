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

"""Map QuAM pulse labels onto Qiskit gate names, including provider-specific gates.

Author: Arthur Strauss
Date: 2026-10-29
"""

from qiskit.circuit import Parameter
from qiskit.circuit.library import get_standard_gate_name_mapping

from ...additional_gates import CRGate, FSimGate, GPI2Gate, GPIGate, SYGate, SYdgGate


def look_for_standard_op(op: str):
    op = op.lower()
    mapping = {
        "cphase": "cp",
        "cnot": "cx",
        "-x/2": "sxdg",
        "-x90": "sxdg",
        "-y/2": "sydg",
        "-y90": "sydg",
        "x/2": "sx",
        "x90": "sx",
        "rx90": "sx",
        "x180": "x",
        "y180": "y",
        "y90": "sy",
        "y/2": "sy",
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
    """
    Returns a dictionary of gate names to standard gate instances, with additional custom gates.
    Custom gates are:
    - SYGate: Rotation around the Y axis by π/2
    - SYdgGate: Rotation around the Y axis by -π/2
    - CRGate: Cross-resonance gate
    - FSimGate: Two-qubit gate parametrized by (θ, ϕ)

    Args:
        None

    Returns:
        A dictionary of gate names to standard gate instances, with additional custom gates.
    """
    gate_map = get_standard_gate_name_mapping()
    gate_map["sy"] = SYGate()
    gate_map["cr"] = CRGate()
    gate_map["sydg"] = SYdgGate()
    gate_map["fsim"] = FSimGate(Parameter("θ"), Parameter("ϕ"))
    gate_map["gpi"] = GPIGate(Parameter("φ"))
    gate_map["gpi2"] = GPI2Gate(Parameter("φ"))

    return gate_map
