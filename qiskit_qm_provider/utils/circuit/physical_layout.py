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

"""Check that a circuit is already laid out on the backend's physical qubits.

Author: Arthur Strauss
Date: 2026-10-29
"""

from qiskit.circuit import QuantumCircuit, QuantumRegister


def ensure_circuit_physical(qc: QuantumCircuit, n_qubits: int) -> None:
    """Validate that ``qc`` is already in canonical physical form for ``n_qubits`` qubits.

    This is the form :meth:`QuantumCircuit.ensure_physical` leaves a circuit in after
    layout: a single owning ``"q"`` register of exactly ``n_qubits`` qubits, plus a
    :class:`~qiskit.transpiler.TranspileLayout` whose physical side covers the same
    ``n_qubits`` qubits. A circuit with no layout is rejected even if its register
    happens to look physical: the OpenQASM 3 lowering needs a circuit that went
    through layout (and translation) against this backend, not merely one of the
    right width. ``qc`` is only inspected, never modified.

    Args:
        qc: QuantumCircuit to check.
        n_qubits: Number of qubits of the backend.

    Raises:
        ValueError: If ``qc`` is not already physical for ``n_qubits`` qubits.
    """
    canonical = QuantumRegister(n_qubits, "q")
    layout = qc.layout
    is_physical = (
        layout is not None
        and qc.num_qubits == n_qubits
        and qc.qregs == [canonical]
        and qc.qubits == list(canonical)
        and len(layout.initial_layout.get_physical_bits()) == n_qubits
        and (layout.final_layout is None or len(layout.final_layout.get_physical_bits()) == n_qubits)
    )
    if not is_physical:
        raise ValueError(
            f"Circuit is not expressed in terms of this backend's {n_qubits} physical qubits. "
            "Transpile it against this backend (or its target) before submitting it for compilation."
        )
