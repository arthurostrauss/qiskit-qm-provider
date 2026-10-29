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

"""Hashable keys for custom gate calibrations.

Author: Arthur Strauss
Date: 2026-10-29
"""

from typing import List

from qiskit.circuit import QuantumCircuit


def operation_key(name: str, number_of_params: int, qubits) -> tuple:
    """A plain, value-hashable stand-in key for ``qm_qasm.OperationIdentifier``.

    ``OperationIdentifier`` defines neither ``__eq__`` nor ``__hash__``, so it
    falls back to Python's default: identity. Two separately-constructed
    ``OperationIdentifier("rz", 1, (0,))`` instances therefore compare unequal
    and hash differently, even though they are logically the same operation
    (verified directly against qm-qasm 1.7.7: ``a == b`` is ``False``,
    ``hash(a) == hash(b)`` is ``False``). Using ``OperationIdentifier``
    instances directly as ``dict``/``set`` keys means a later insertion for
    what looks like the same operation never overwrites/dedupes the earlier
    one -- it silently adds a second, functionally duplicate entry instead.
    That is a real, previously-unnoticed bug (see ``has_conflicting_calibrations``
    and ``QMBackend``'s ``_operation_mapping_QUA``/``_calibration_operation_mapping_QUA``
    below, both fixed by keying on this instead).

    This tuple has ordinary Python value equality, so the same logical
    operation always produces the same key. Build the real
    ``OperationIdentifier`` objects ``HardwareConfig`` needs from this key only
    once, right before compiling -- see ``QMBackend.compiler``.
    """
    return (name, number_of_params, tuple(qubits))


def has_conflicting_calibrations(circuits: List[QuantumCircuit]) -> bool:
    """Check whether circuits define conflicting custom calibrations.

    Args:
        circuits: Circuits whose ``calibrations`` attributes are checked.

    Returns:
        ``True`` if the same operation identifier appears more than once.
    """
    custom_gates = set()
    for qc in circuits:
        if hasattr(qc, "calibrations") and qc.calibrations:
            for gate_name, cal_info in qc.calibrations.items():
                for qubits, parameters in cal_info.keys():
                    op_id = operation_key(gate_name, len(parameters), qubits)
                    if op_id not in custom_gates:
                        custom_gates.add(op_id)
                    else:
                        return True
    return False
