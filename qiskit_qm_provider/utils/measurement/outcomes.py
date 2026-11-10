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

"""Wire classical measurement outcomes from an embedded circuit into QUA variables.

Author: Arthur Strauss
Date: 2026-10-29
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit import QuantumCircuit
from quam.utils.qua_types import QuaVariableInt

if TYPE_CHECKING:
    from qm_qasm import CompilationResult


def get_measurement_outcomes(
    qc: QuantumCircuit, result: CompilationResult, compute_state_int: bool = True
) -> dict[str, dict[str, QuaVariableInt]]:
    """Build a dict view of wired measurement handles (same data as ``comp.outputs``).

    Prefer :attr:`~qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.outputs`
    on the object returned by :meth:`~qiskit_qm_provider.backend.QMBackend.quantum_circuit_to_qua`.
    This helper remains for older call sites that expect a nested dict.

    Call inside the same ``with program():`` block **immediately after**
    :meth:`~.QMBackend.quantum_circuit_to_qua`. The returned QUA variables reference
    outcomes from the circuit execution that just completed, enabling real-time QUA
    control flow and streaming without round-tripping through Python.

    Args:
        qc: The :class:`~qiskit.circuit.QuantumCircuit` that was compiled.
        result: The compilation result returned by ``quantum_circuit_to_qua``, or a
            :class:`~qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation`
            wrapper.
        compute_state_int: If ``True`` (default), declare an integer packing of
            each register's bits (LSB = qubit index 0).

    Returns:
        A dictionary mapping each output key to a sub-dictionary:

        - ``"value"``: QUA variable for the output (a bool array for multi-bit
          classical registers, a bool scalar for loose clbits).
        - ``"is_array"``: ``True`` when ``"value"`` is a QUA array, ``False`` for a scalar
          — lets callers choose ``value[i]`` vs ``value`` when saving. Mirrors
          ``Parameter.is_array``.
        - ``"length"``: ``Parameter`` convention — ``0`` for a scalar output (loose clbit),
          otherwise the register's bit count.
        - ``"state_int"``: QUA ``int`` with packed bit values, LSB = bit 0 (when
          ``compute_state_int=True``).
        - ``"stream"``: QUA stream for ``stream_processing()`` on the host.

        Each classical register appears under its own name. Loose clbits not in any
        register appear under their own per-bit keys ``_bit0``, ``_bit1``, … (one entry
        per bit, never packed into a single register) — the same keys the
        :attr:`~qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.outputs`
        table exposes. All entries are sourced from that table, so ``state_int`` is the
        cached, rewire-aware handle (``meas[key]["state_int"] ≡ comp.outputs.state_ints[key]``).
    """
    from ...backend.qua_circuit_compilation import MeasurementOutcomeTable, QuaCircuitCompilation, _loose_bit_keys

    if isinstance(result, QuaCircuitCompilation):
        table = result.outputs
    else:
        table = MeasurementOutcomeTable.from_compilation(qc, result)

    def _entry(key: str) -> dict:
        field = table.get_parameter(key)
        entry = {
            "value": field.var,
            "is_array": field.is_array,
            "stream": field.stream,
            "length": field.length,
        }
        if compute_state_int:
            entry["state_int"] = field.state_int
        return entry

    clbits_dict: dict[str, dict] = {creg.name: _entry(creg.name) for creg in qc.cregs}
    # Loose clbits are independent single bits, not a register — expose one entry each.
    for key in _loose_bit_keys(qc):
        clbits_dict[key] = _entry(key)

    return clbits_dict
