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
* :func:`export_qasm3_with_input_names` -- export a circuit and resolve the identifier the
  OpenQASM 3 exporter actually gave each input (it may rename on clashes).
* :func:`validate_classical_types` -- reject classical types QUA cannot represent.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, Iterator, List, Tuple

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
    """Provider-side table name for a Qiskit input object.

    ``ParameterVector`` -> vector name (array mode), ``ParameterVectorElement`` ->
    ``_name_i_`` (per-element mode), ``Parameter``/``Var`` -> their name.
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


def for_loop_parameters(qc: "QuantumCircuit") -> set:
    """Return the set of loop parameters used by ``for`` loops anywhere in ``qc``."""
    from qiskit.circuit import ForLoopOp

    params = set()
    for circuit in _iter_circuits(qc):
        for instruction in circuit.data:
            op = instruction.operation
            if isinstance(op, ForLoopOp) and op.params[1] is not None:
                params.add(op.params[1])
    return params


# --------------------------------------------------------------------------------------------
# Exported input names
# --------------------------------------------------------------------------------------------

_INPUT_DECL = re.compile(r"^\s*input\s+.*?\s(\w+)\s*;\s*$", re.MULTILINE)


def declared_input_names(qasm: str) -> List[str]:
    """Identifiers of all ``input ...;`` declarations in an OpenQASM 3 program, in order."""
    return _INPUT_DECL.findall(qasm)


def _input_objects(qc: "QuantumCircuit") -> list:
    """Circuit objects that may become ``input`` declarations, in exporter order."""
    from qiskit.circuit.parametervector import ParameterVectorElement

    array_mode = qiskit_has_array()
    objects, seen = [], set()
    for parameter in qc.parameters:
        obj = parameter
        if array_mode and isinstance(parameter, ParameterVectorElement):
            obj = parameter.vector
            if obj.name in seen:
                continue
            seen.add(obj.name)
        objects.append(obj)
    if hasattr(qc, "iter_input_vars"):
        objects.extend(qc.iter_input_vars())
    return objects


def _symbol_table_names(exporter, qc) -> Tuple[str, Dict[str, str]]:
    """Export via ``QASM3Builder`` (private API) and read identifiers from its symbol table."""
    import io

    from qiskit.qasm3.exporter import QASM3Builder
    from qiskit.qasm3.printer import BasicPrinter

    builder = QASM3Builder(
        qc,
        includeslist=exporter.includes,
        basis_gates=exporter.basis_gates,
        disable_constants=exporter.disable_constants,
        allow_aliasing=exporter.allow_aliasing,
        experimental=exporter.experimental,
        annotation_handlers=exporter.annotation_handlers,
        implicit_defcals=exporter.implicit_defcals,
    )
    program = builder.build_program()
    with io.StringIO() as stream:
        BasicPrinter(stream, indent=exporter.indent, experimental=exporter.experimental).visit(program)
        qasm = stream.getvalue()
    global_objects = builder.symbols.objects[0]
    names = {}
    for obj in _input_objects(qc):
        ident = global_objects.get(obj)
        if ident is not None and type(ident).__name__ == "Identifier":
            names[qiskit_input_key(obj)] = ident.string
    return qasm, names


def export_qasm3_with_input_names(exporter, qc: "QuantumCircuit") -> Tuple[str, Dict[str, str]]:
    """Export ``qc`` with ``exporter`` and map provider input keys to exported identifiers.

    The exporter registers inputs with ``allow_rename=True``, so a vector called ``x`` (a gate
    name) becomes e.g. ``x_1``. Names are read from the exporter's symbol table; if that private
    API is unavailable, the emitted ``input`` declarations are parsed and matched by name, then
    by order.

    Returns:
        ``(qasm, {qiskit_input_key(obj): exported_identifier})`` for every exported input.

    Raises:
        ValueError: if the exported input names cannot be matched unambiguously.
    """
    objects = _input_objects(qc)
    try:
        qasm, names = _symbol_table_names(exporter, qc)
    except (ImportError, AttributeError, TypeError):
        qasm, names = exporter.dumps(qc), None
    declared = declared_input_names(qasm)
    declared_set = set(declared)

    mapping: Dict[str, str] = {}
    if names is not None:
        mapping = {key: name for key, name in names.items() if name in declared_set}
        return qasm, mapping

    # Fallback: parse declarations. Exact matches first, then pair the rest by order.
    keys = [qiskit_input_key(obj) for obj in objects]
    exact = [k for k in keys if k in declared_set]
    mapping.update({k: k for k in exact})
    rest_keys = [k for k in keys if k not in declared_set]
    rest_declared = [d for d in declared if d not in mapping.values()]
    if rest_declared:
        if len(rest_keys) != len(rest_declared):
            raise ValueError(
                "Cannot match exported OpenQASM 3 inputs "
                f"{rest_declared} to circuit inputs {rest_keys} unambiguously."
            )
        mapping.update(zip(rest_keys, rest_declared))
    return qasm, mapping
