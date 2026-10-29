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

* :func:`qiskit_has_array` -- classical array support (``types.Array``, i.e. the
  ``arthurostrauss/qiskit@full_real_time_expr`` fork).
* :func:`qiskit_supports_vector_arrays` -- whether the exporter can emit a ``ParameterVector`` as
  one ``input`` array, which decides the table layout of vectors.
* :func:`qiskit_input_key` -- the name the provider uses for a circuit input in a
  :class:`~qiskit_qm_provider.parameter_table.ParameterTable`.
* :func:`bind_exported_inputs` -- map table fields onto the inputs the exporter actually declared.
* :func:`validate_classical_types` -- reject classical types QUA cannot represent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, Mapping

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

VECTOR_ARRAYS_FEATURE = "PARAMETER_VECTORS_AS_ARRAYS"
"""The ``qiskit.qasm3.ExperimentalFeatures`` flag that exports vectors as ``input`` arrays."""


def qiskit_has_array() -> bool:
    """Return ``True`` iff the installed Qiskit provides classical arrays (``types.Array``)."""
    from qiskit.circuit.classical import types

    return hasattr(types, "Array")


def qiskit_supports_vector_arrays() -> bool:
    """Return ``True`` iff the installed Qiskit can export a ``ParameterVector`` as one
    full-length ``input array[float[64], N]`` (the ``PARAMETER_VECTORS_AS_ARRAYS``
    experimental exporter feature of the ``full_real_time_expr`` fork).

    This decides the layout :meth:`~qiskit_qm_provider.parameter_table.ParameterTable.from_qiskit`
    gives vectors: one array field when supported, one scalar field per used element otherwise.
    Whether a backend actually exports vectors as arrays is its ``export_vector_as_array``
    option; an array field binds either export (see :func:`bind_exported_inputs`).
    """
    from qiskit.qasm3 import ExperimentalFeatures

    return hasattr(ExperimentalFeatures, VECTOR_ARRAYS_FEATURE)


def vector_element_key(parameter) -> str:
    """Field name of a single ``ParameterVectorElement``: ``_name_i_``, the identifier Qiskit
    exports for ``name[i]`` as a scalar input."""
    return f"_{parameter.vector.name}_{parameter.index}_"


def qiskit_input_key(obj) -> str:
    """Name of the table field holding a circuit input, in the layout of
    :meth:`~qiskit_qm_provider.parameter_table.ParameterTable.from_qiskit`: a vector element maps to
    its vector's array field when :func:`qiskit_supports_vector_arrays`, else to its own field."""
    from qiskit.circuit.parametervector import ParameterVectorElement

    if isinstance(obj, ParameterVectorElement):
        if qiskit_supports_vector_arrays():
            return obj.vector.name
        return vector_element_key(obj)
    return obj.name


def validate_qiskit_input_names(qc: QuantumCircuit) -> None:
    """Reject ambiguous input names before a table drops one of them.

    Qiskit permits a scalar ``theta`` alongside ``theta[0]``, but with vector arrays both
    would become a field named ``theta``. An element's own key ``_theta_0_`` is reserved too
    (it is how a single element is bound when exported as a scalar input), so a scalar named
    ``_theta_0_`` is rejected as well. Pulse schedules (no input vars) are checked on their
    parameters only.
    """
    from qiskit.circuit.parametervector import ParameterVectorElement

    vector_arrays = qiskit_supports_vector_arrays()
    names = {}
    for obj in (*qc.parameters, *getattr(qc, "iter_input_vars", tuple)()):
        element = isinstance(obj, ParameterVectorElement)
        owner = obj.vector if vector_arrays and element else obj
        keys = {qiskit_input_key(obj), vector_element_key(obj)} if element else {qiskit_input_key(obj)}
        for name in keys:
            if name in names and names[name] != owner:
                raise ValueError(
                    f"Ambiguous Qiskit input name '{name}': distinct inputs map to the same field. "
                    "Rename the scalar, ParameterVector or classical input to use distinct names."
                )
            names[name] = owner


_UNKNOWN = object()


def _qua_length(value):
    """Array length of a QUA value bound to an input, ``None`` for a scalar, or ``_UNKNOWN``."""
    from qm.qua._expressions import QuaArrayVariable

    if isinstance(value, QuaArrayVariable):
        size = getattr(value, "_size", None)
        return size if isinstance(size, int) else _UNKNOWN
    if hasattr(value, "__len__"):  # a QUA2DArray row
        return len(value)
    return None


def _shape(length) -> str:
    return "a scalar" if length is None else f"an array of length {length}"


def _input_label(obj) -> str:
    from qiskit.circuit import ParameterVector

    kind = "ParameterVector" if isinstance(obj, ParameterVector) else type(obj).__name__
    return f"{kind} '{obj.name}'"


def bind_exported_inputs(exported: Mapping[Any, Any], bindings: Mapping[str, Any]) -> Dict[str, Any]:
    """Map QUA values onto the ``input`` declarations of an exported circuit.

    The exporter may rename an input (OpenQASM 3 gives gates and variables one namespace, so a
    vector ``x`` next to an ``x`` gate becomes ``x_0``), so inputs are matched by Qiskit object,
    never by predicting identifiers.

    Args:
        exported: Each declared input's Qiskit object mapped to its
            :class:`~qiskit_qm_provider.qasm3_exporter.ExportedInput`.
        bindings: QUA values keyed by table field name (:func:`qiskit_input_key` layout, or
            :func:`vector_element_key` for a single element). A value may also be keyed by the
            exported identifier itself. Keys matching no input are ignored, so a table shared by
            several circuits can be passed whole.

    Returns:
        The QUA value for each exported identifier, ready for the qm-qasm compiler.

    Raises:
        ValueError: listing every input with no binding or with a binding of the wrong shape
            (qm-qasm would otherwise fail opaquely, or silently broadcast a scalar to an array).
    """
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.parametervector import ParameterVectorElement

    inputs, problems = {}, []
    for obj, spec in exported.items():
        value, hint = _UNKNOWN, ""
        if isinstance(obj, ParameterVector):  # exported as one input array
            value = bindings.get(obj.name, _UNKNOWN)
            if value is _UNKNOWN and any(vector_element_key(p) in bindings for p in obj):
                hint = (
                    " The bindings hold separate element fields; bind the whole vector with a "
                    f"length-{len(obj)} array, or set the backend option export_vector_as_array=False."
                )
        elif isinstance(obj, ParameterVectorElement):  # exported as a scalar input
            value = bindings.get(vector_element_key(obj), _UNKNOWN)
            array = bindings.get(obj.vector.name, _UNKNOWN)
            if value is _UNKNOWN and array is not _UNKNOWN and _qua_length(array) in (len(obj.vector), _UNKNOWN):
                value = array[obj.index]  # an array field binds the element's cell
        else:
            value = bindings.get(obj.name, _UNKNOWN)
        if value is _UNKNOWN:
            value = bindings.get(spec.name, _UNKNOWN)
        label = f"{_input_label(obj)} (exported as '{spec.name}', {_shape(spec.length)})"
        if value is _UNKNOWN:
            problems.append(f"{label} has no binding.{hint}")
            continue
        length = _qua_length(value)
        if length is not _UNKNOWN and length != spec.length:
            problems.append(f"{label} is bound to {_shape(length)}.")
            continue
        inputs[spec.name] = value
    if problems:
        raise ValueError("Cannot bind the circuit inputs:\n- " + "\n- ".join(problems))
    return inputs


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

    vector_arrays = qiskit_supports_vector_arrays()
    fields = {p.name: p for p in table.parameters}
    layout = []
    for column, parameter in enumerate(qc.parameters):
        key = qiskit_input_key(parameter)
        field = fields.get(key)
        if field is None:
            continue
        if vector_arrays and isinstance(parameter, ParameterVectorElement):
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
