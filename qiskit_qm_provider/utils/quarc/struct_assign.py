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

"""Copy a declared ParameterTable into a matching Quarc struct.

Author: Arthur Strauss
Date: 2026-10-29
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from qm.qua import assign

if TYPE_CHECKING:
    from ...parameter_table import ParameterTable


def _require_qua_struct_handle(struct: Any) -> Any:
    """Return ``struct`` after validating it is a Quarc :class:`QuaStructHandle`."""
    try:
        from quarc.dsl.structs.qua_struct_handle import QuaStructHandle
    except ImportError as exc:
        raise ImportError("assign_struct_with_table requires the `quarc` package. Install `quarc` to use it.") from exc
    if not isinstance(struct, QuaStructHandle):
        raise TypeError(
            "struct must be a quarc QuaStructHandle (from module.add_struct). " f"Got {type(struct).__name__}."
        )
    return struct


def _struct_field_specs(struct: Any) -> Dict[str, Dict[str, Any]]:
    """Return ``{field_name: {is_array, length}}`` for a QuaStructHandle."""
    from typing import get_args, get_origin, get_type_hints

    from quarc import Array, Scalar

    annotations = get_type_hints(struct._struct_spec.struct)
    specs: Dict[str, Dict[str, Any]] = {}
    for field_name, annotation in annotations.items():
        origin = get_origin(annotation)
        if origin is Scalar:
            specs[field_name] = {"is_array": False, "length": 0}
        elif origin is Array:
            args = get_args(annotation)
            specs[field_name] = {"is_array": True, "length": args[1]}
        else:
            raise TypeError(f"Struct field {field_name!r} has unsupported Quarc annotation {annotation!r}.")
    return specs


def _validate_struct_table_match(struct: Any, table: "ParameterTable") -> None:
    struct_fields = _struct_field_specs(struct)
    table_fields = {parameter.name: parameter for parameter in table.parameters}

    if set(struct_fields) != set(table_fields):
        missing_in_table = sorted(set(struct_fields) - set(table_fields))
        missing_in_struct = sorted(set(table_fields) - set(struct_fields))
        details = []
        if missing_in_table:
            details.append(f"missing from ParameterTable: {missing_in_table}")
        if missing_in_struct:
            details.append(f"missing from struct: {missing_in_struct}")
        raise ValueError(
            "Struct fields and ParameterTable parameters must have exactly the same names. " + "; ".join(details)
        )

    for field_name, spec in struct_fields.items():
        parameter = table_fields[field_name]
        if parameter.is_array != spec["is_array"]:
            raise ValueError(
                f"Field {field_name!r}: struct/table shape mismatch "
                f"(struct is_array={spec['is_array']}, parameter is_array={parameter.is_array})."
            )
        if spec["is_array"] and parameter.length != spec["length"]:
            raise ValueError(
                f"Field {field_name!r}: struct array length {spec['length']} does not match "
                f"parameter length {parameter.length}."
            )
        if not parameter.is_declared:
            raise ValueError(
                f"Parameter {field_name!r} is not declared in QUA. Call "
                f"ParameterTable.declare() (or initialize the owning OPNIC table) first."
            )

    if struct.qua_struct is None:
        raise ValueError("Struct is not initialized in QUA. Call QuaStructHandle.initialize_in_qua() first.")


def assign_struct_with_table(struct: Any, table: "ParameterTable") -> None:
    """QUA macro: copy declared parameter values into a matching OPNIC struct.

    Call inside the same ``with program():`` block after both sides are ready:
    the :class:`~.ParameterTable` parameters must already be declared (via
    :meth:`~.ParameterTable.declare` or OPNIC
    :meth:`~.ParameterTable.initialize_in_qua`), and ``struct`` must be a Quarc
    **QuaStructHandle** returned by ``module.add_struct(...)`` with
    :meth:`QuaStructHandle.initialize_in_qua` already invoked.

    For each field, this macro assigns the table parameter's QUA variable onto
    the corresponding struct field using ``qm.qua.assign``. Field names and
    shapes (scalar vs. array length) must match exactly between the struct spec
    and the table.

    Args:
        struct: A Quarc ``QuaStructHandle`` bound to the destination OPNIC struct.
            The type is validated at runtime via a lazy ``quarc`` import; it is
            not imported for static type checking in this module.
        table: Source :class:`~.ParameterTable` whose declared QUA variables are
            copied field-by-field into ``struct``.

    Raises:
        ImportError: If ``quarc`` is not installed.
        TypeError: If ``struct`` is not a Quarc ``QuaStructHandle``.
        ValueError: If field names or sizes differ, parameters are undeclared,
            or ``struct`` was not initialized in QUA.
    """
    from ...parameter_table import ParameterTable

    if not isinstance(table, ParameterTable):
        raise TypeError(f"table must be a ParameterTable, got {type(table).__name__}.")

    struct = _require_qua_struct_handle(struct)
    _validate_struct_table_match(struct, table)

    for field_name in _struct_field_specs(struct):
        parameter = table.table[field_name]
        struct_field = getattr(struct.qua_struct, field_name)
        if parameter.is_array:
            for index in range(parameter.length):
                assign(struct_field[index], parameter.var[index])
        else:
            assign(struct_field[0], parameter.var)
