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

"""Helpers bridging Qiskit classical data (parameters, ``Var`` s, arrays) and the provider.

* :func:`qiskit_has_array` -- array mode detection (``types.Array`` present, i.e. the
  ``arthurostrauss/qiskit@full_real_time_expr`` fork).
* :func:`qiskit_input_key` -- the name the provider uses for a circuit input in a
  :class:`~qiskit_qm_provider.parameter_table.ParameterTable`.
* :func:`validate_classical_types` -- reject classical types QUA cannot represent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def qiskit_has_array() -> bool:
    """Return ``True`` iff the installed Qiskit provides ``types.Array`` (array mode).

    In array mode the OpenQASM 3 exporter emits every ``ParameterVector`` as one
    full-length ``input array[float[64], N]``; otherwise each used element is exported
    as its own scalar input.
    """
    from qiskit.circuit.classical import types

    return hasattr(types, "Array")


def qiskit_input_key(obj) -> str:
    """Provider-side input name: ``_name_i_`` for a ``ParameterVectorElement`` (the identifier
    stock Qiskit's OpenQASM 3 exporter emits for ``name[i]``), otherwise ``obj.name``.
    """
    from qiskit.circuit.parametervector import ParameterVectorElement

    if isinstance(obj, ParameterVectorElement):
        return f"_{obj.vector.name}_{obj.index}_"
    return obj.name


# --------------------------------------------------------------------------------------------
# Unsupported classical types
# --------------------------------------------------------------------------------------------


def _unsupported_reason(ty) -> str | None:
    from qiskit.circuit.classical import types

    kind = ty.kind
    if kind in (getattr(types, "Duration", None), getattr(types, "Stretch", None)):
        return (
            "Duration/Stretch values are compile-time only in QUA; use them in delays "
            "(e.g. qc.delay(expr) / qc.add_stretch) instead of as variables, stores, inputs or arrays"
        )
    array_cls = getattr(types, "Array", None)
    if array_cls is not None and kind is array_cls:
        if ty.size == 0:
            return "zero-size arrays cannot be declared in QUA"
        return _unsupported_reason(ty.element)
    return None


def unsupported_classical_type_error(name: str, ty) -> ValueError | None:
    """Return the provider's (single, consistent) error for an unsupported type, or ``None``."""
    reason = _unsupported_reason(ty)
    if reason is None:
        return None
    return ValueError(f"Unsupported classical type {ty} for variable '{name}': {reason}.")


def _iter_circuits(qc: "QuantumCircuit") -> Iterator["QuantumCircuit"]:
    yield qc
    for instruction in qc.data:
        for block in getattr(instruction.operation, "blocks", ()):
            yield from _iter_circuits(block)


def validate_classical_types(qc: "QuantumCircuit") -> None:
    """Raise ``ValueError`` if ``qc`` (or a nested control-flow block) declares, captures or
    takes as input a ``Var`` of a type QUA cannot represent (Duration/Stretch, zero-size arrays,
    Duration arrays). Stores into such variables are covered since their target is such a Var.
    """
    for circuit in _iter_circuits(qc):
        if not hasattr(circuit, "iter_vars"):
            continue
        for var in circuit.iter_vars():
            error = unsupported_classical_type_error(var.name, var.type)
            if error is not None:
                raise error
