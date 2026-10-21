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
    """Input field name, following the installed Qiskit's vector export layout."""
    from qiskit.circuit.parametervector import ParameterVectorElement

    if isinstance(obj, ParameterVectorElement):
        if qiskit_has_array():
            return obj.vector.name
        return f"_{obj.vector.name}_{obj.index}_"
    return obj.name


def validate_qiskit_input_names(qc: QuantumCircuit) -> None:
    """Reject ambiguous input names before the exporter renames or a table drops them.

    Qiskit permits a scalar ``theta`` alongside ``theta[0]``, but in array mode both
    would become a field named ``theta``. The provider requires distinct input names.
    Pulse schedules (no input vars) are checked on their parameters only.
    """
    from qiskit.circuit.parametervector import ParameterVectorElement

    array_mode = qiskit_has_array()
    names = {}
    for obj in (*qc.parameters, *getattr(qc, "iter_input_vars", tuple)()):
        owner = obj.vector if array_mode and isinstance(obj, ParameterVectorElement) else obj
        name = qiskit_input_key(obj)
        if name in names and names[name] != owner:
            raise ValueError(
                f"Ambiguous Qiskit input name '{name}': distinct inputs map to the same field. "
                "Rename the scalar, ParameterVector or classical input to use distinct names."
            )
        names[name] = owner


def qua_zero(qua_type):
    """Zero value of a table field's QUA type (``int``, ``bool`` or ``fixed``)."""
    return {int: 0, bool: False}.get(qua_type, 0.0)


def qiskit_row_layout(qc, table) -> list[tuple[int, str, int | None]]:
    """``(column, field, index)`` for each ``qc.parameters`` column bound to a ``table`` field.

    ``index`` is the element index for a vector packed into an array field (array mode),
    ``None`` for a scalar field. Raises ``ValueError`` if a vector's field is not an array
    of the vector's length.
    """
    from qiskit.circuit.parametervector import ParameterVectorElement

    array_mode = qiskit_has_array()
    fields = {p.name: p for p in table.parameters}
    layout = []
    for column, parameter in enumerate(qc.parameters):
        key = qiskit_input_key(parameter)
        field = fields.get(key)
        if field is None:
            continue
        if array_mode and isinstance(parameter, ParameterVectorElement):
            if not field.is_array or field.length != len(parameter.vector):
                raise ValueError(
                    f"Table field '{key}' must be an array of length {len(parameter.vector)} "
                    f"to hold ParameterVector '{parameter.vector.name}'."
                )
            layout.append((column, key, parameter.index))
        else:
            layout.append((column, key, None))
    return layout


def qiskit_parameter_packer(qc, table):
    """Return ``row -> dict`` packing a row ordered as ``qc.parameters`` into table fields.

    The column layout is computed (and validated) once, so build the packer outside
    per-row loops. Unused vector elements and fields the row does not bind are filled
    with typed zeros: the fork exports the full vector even though PUBs contain only
    the circuit's used parameters.
    """
    layout = qiskit_row_layout(qc, table)
    defaults = {
        p.name: [qua_zero(p.type)] * p.length if p.is_array else qua_zero(p.type) for p in table.parameters
    }

    def pack(row) -> dict:
        values = {name: list(value) if isinstance(value, list) else value for name, value in defaults.items()}
        for column, key, index in layout:
            if index is None:
                values[key] = row[column]
            else:
                values[key][index] = row[column]
        return values

    return pack


def packed_qiskit_parameter_rows(pub, table) -> list[list]:
    """Return PUB bindings packed in table order, suitable for a cloud sync hook."""
    pack = qiskit_parameter_packer(pub.circuit, table)
    return [
        list(pack(row).values()) for row in pub.parameter_values.ravel().as_array(pub.circuit.parameters).tolist()
    ]


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


def _iter_circuits(qc: QuantumCircuit) -> Iterator[QuantumCircuit]:
    if not hasattr(qc, "iter_vars"):  # pulse schedules carry no classical Vars
        return
    yield qc
    for instruction in qc.data:
        for block in getattr(instruction.operation, "blocks", ()):
            yield from _iter_circuits(block)


def validate_classical_types(qc: QuantumCircuit) -> None:
    """Raise ``ValueError`` if ``qc`` (or a nested control-flow block) declares, captures or
    takes as input a ``Var`` of a type QUA cannot represent (Duration/Stretch, zero-size arrays,
    Duration arrays). Stores into such variables are covered since their target is such a Var.
    """
    for circuit in _iter_circuits(qc):
        for var in circuit.iter_vars():
            error = unsupported_classical_type_error(var.name, var.type)
            if error is not None:
                raise error
