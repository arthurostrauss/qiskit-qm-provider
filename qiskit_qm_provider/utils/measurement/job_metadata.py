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

"""Bit widths and Qiskit experiment headers for classified measurement results.

Author: Arthur Strauss
Date: 2026-10-29
"""

from __future__ import annotations

from typing import Any

from qiskit.circuit import QuantumCircuit


def measurement_output_bit_sizes(qc: QuantumCircuit) -> dict[str, int]:
    """Map measurement stream keys to bit width for classified result assembly.

    Classical registers use their creg name and width. Loose clbits (not in any
    register) each appear under ``_bit0``, ``_bit1``, … with width ``1``, matching
    :func:`~qiskit_qm_provider.backend.qua_circuit_compilation._loose_bit_keys` and
    QUA stream names in :func:`~qiskit_qm_provider.job.qua_programs.get_run_program`.
    """
    from ...backend.qua_circuit_compilation import _loose_bit_keys

    sizes = {creg.name: creg.size for creg in qc.cregs}
    for key in _loose_bit_keys(qc):
        sizes[key] = 1
    return sizes


def experiment_result_header(qc: QuantumCircuit) -> dict[str, Any]:
    """Build a Qiskit :class:`~qiskit.result.models.ExperimentResult` header for *qc*.

    One circuit maps to one experiment result in the legacy ``Result`` API.  The
    header fields mirror reference simulators (e.g. ``BasicSimulator``) so
    :meth:`~qiskit.result.Result.get_counts`, :meth:`~qiskit.result.Result.get_memory`,
    and name-based experiment lookup via :meth:`~qiskit.result.Result.data` work as
    expected.

    ``creg_sizes`` follows :func:`measurement_output_bit_sizes` key order (classical
    registers first, then loose clbits as ``_bitN`` entries of width ``1``).  That
    matches the bit order produced when classified measurement streams are joined in
    :meth:`~qiskit_qm_provider.job.qm_job.QMJob._build_result_function`.
    """
    from ...backend.qua_circuit_compilation import _loose_bit_keys

    output_sizes = measurement_output_bit_sizes(qc)
    creg_sizes = [[name, size] for name, size in output_sizes.items()]
    memory_slots = sum(output_sizes.values())

    clbit_labels = [[creg.name, j] for creg in qc.cregs for j in range(creg.size)]
    for key in _loose_bit_keys(qc):
        clbit_labels.append([key, 0])

    return {
        "name": qc.name,
        "n_qubits": qc.num_qubits,
        "qreg_sizes": [[qreg.name, qreg.size] for qreg in qc.qregs],
        "creg_sizes": creg_sizes,
        "qubit_labels": [[qreg.name, j] for qreg in qc.qregs for j in range(qreg.size)],
        "clbit_labels": clbit_labels,
        "memory_slots": memory_slots,
        "global_phase": qc.global_phase,
        "metadata": qc.metadata if qc.metadata is not None else {},
    }
