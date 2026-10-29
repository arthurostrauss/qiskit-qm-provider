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

"""Validate a QuAM root before the backend uses it.

Author: Arthur Strauss
Date: 2026-10-29
"""

from quam.components import Qubit, QubitPair
from quam.core import QuamRoot


def validate_machine(machine) -> QuamRoot:
    """Validate a QuAM instance before use with the backend.

    Args:
        machine: QuAM instance to validate. Must expose ``qubits`` and
            ``qubit_pairs`` containing :class:`~quam.components.Qubit` and
            :class:`~quam.components.QubitPair` objects respectively.

    Returns:
        The validated QuAM instance.

    Raises:
        ValueError: If required attributes are missing or have wrong types.
    """
    if not hasattr(machine, "qubits") or not hasattr(machine, "qubit_pairs"):
        raise ValueError("Invalid QuAM instance provided, should have qubits and qubit_pairs attributes")
    if not all(isinstance(qubit, Qubit) for qubit in machine.qubits.values()):
        raise ValueError("All qubits should be of type Qubit")
    if not all(isinstance(qubit_pair, QubitPair) for qubit_pair in machine.qubit_pairs.values()):
        raise ValueError("All qubit pairs should be of type QubitPair")

    return machine
