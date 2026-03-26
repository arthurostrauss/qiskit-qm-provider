# Copyright 2026 Arthur Strauss
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build an in-process Quarc ``BaseModule`` from DGX packet targets and sync stream ids to pool objects.

Requires the ``quarc`` package. Mirrors ``quarc build`` config import: struct registration uses
``StructSpec.from_struct`` / Quarc's global stream id counters (isolated here when building).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Sequence, TYPE_CHECKING, Type, Union

from qm.qua import fixed

from .quarc_naming import default_quarc_struct_name

if TYPE_CHECKING:
    from .parameter import Parameter
    from .parameter_table import ParameterTable

SPECS_VERSION = 1

# Default Python subclass name for :class:`quarc.BaseModule` when the user does not specify one.
DEFAULT_QUARC_MODULE_CLASS_NAME = "QuarcPacketModule"
# Default ``name`` in ``quarc.toml`` when writing a module tree.
DEFAULT_QUARC_MANIFEST_NAME = "quarc_packet_module"


def _field_dtype_for_parameter(param: Any) -> str:
    t = param.type
    if t == bool:
        return "bool"
    if t == int:
        return "int"
    if t == fixed:
        return "float"
    return "float"


def _field_length_for_parameter(param: Any) -> int:
    if getattr(param, "is_array", False):
        return int(param.length)
    return 1


def struct_spec_from_standalone_parameter(param: "Parameter") -> Dict[str, Any]:
    # Private/internal hook used to set standalone DGX packet stream ids.
    param._materialize_dgx_stream_id_for_quarc()
    fields = [
        {
            "name": param.name,
            "dtype": _field_dtype_for_parameter(param),
            "length": _field_length_for_parameter(param),
        }
    ]
    d = param.direction
    return {
        "struct_name": default_quarc_struct_name(param),
        "direction": d.name,
        "fields": fields,
    }


def struct_spec_from_table(table: "ParameterTable") -> Dict[str, Any]:
    fields: List[Dict[str, Any]] = []
    for p in table.parameters:
        fields.append(
            {
                "name": p.name,
                "dtype": _field_dtype_for_parameter(p),
                "length": _field_length_for_parameter(p),
            }
        )
    d = table.direction
    return {
        "struct_name": default_quarc_struct_name(table),
        "direction": d.name,
        "fields": fields,
    }


def specs_from_quarc_packet_targets(
    targets: Sequence[Union["ParameterTable", "Parameter"]],
    *,
    module_class_name: str | None = None,
) -> Dict[str, Any]:
    from .parameter import Parameter
    from .parameter_table import ParameterTable

    structs: List[Dict[str, Any]] = []
    for obj in targets:
        if isinstance(obj, ParameterTable):
            structs.append(struct_spec_from_table(obj))
        elif isinstance(obj, Parameter):
            structs.append(struct_spec_from_standalone_parameter(obj))
        else:
            raise TypeError(f"Expected ParameterTable or Parameter, got {type(obj)!r}")
    out: Dict[str, Any] = {"version": SPECS_VERSION, "structs": structs}
    if module_class_name is not None:
        out["module_class_name"] = module_class_name
    return out


def _dtype_map() -> Dict[str, Type[Any]]:
    return {"float": float, "int": int, "bool": bool}


def _make_field(dtype: str, length: int) -> Any:
    from quarc import Array, Scalar

    d = _dtype_map()[dtype]
    if length <= 1:
        return Scalar[d]
    return Array[d, length]


def _build_struct_class(spec: Dict[str, Any]) -> Any:
    from quarc import Struct

    kwargs = {f["name"]: _make_field(f["dtype"], int(f["length"])) for f in spec["fields"]}
    return Struct(struct_name=spec["struct_name"], **kwargs)


def _resolve_module_class_name(specs: Dict[str, Any]) -> str:
    name = specs.get("module_class_name") or DEFAULT_QUARC_MODULE_CLASS_NAME
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"module_class_name must be a valid Python identifier, got {name!r} "
            f"(set it in specs or pass module_class_name=... to specs_from_quarc_packet_targets)."
        )
    return name


def _make_base_module_class(structs: List[Dict[str, Any]], class_name: str) -> type:
    """Dynamic ``BaseModule`` subclass with the user's chosen ``class_name``."""
    from quarc import BaseModule, Direction

    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        BaseModule.__init__(self)
        for st in structs:
            struct_cls = _build_struct_class(st)
            quarc_dir = getattr(Direction, st["direction"])
            self.add_struct(struct_cls, quarc_dir)

    return type(class_name, (BaseModule,), {"__init__": __init__, "__module__": __name__})


def build_quarc_base_module_from_specs(specs: Dict[str, Any]) -> Any:
    """Instantiate Quarc ``BaseModule`` with ``add_struct`` for each entry (same as generated ``config.py``)."""
    if specs.get("version") != SPECS_VERSION:
        raise ValueError(f"Unsupported specs version {specs.get('version')!r}, expected {SPECS_VERSION}")

    cls_name = _resolve_module_class_name(specs)
    ModuleCls = _make_base_module_class(specs["structs"], cls_name)
    return ModuleCls()


def load_quarc_base_module_from_json_path(specs_path: Path) -> Any:
    p = Path(specs_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return build_quarc_base_module_from_specs(data)


@contextmanager
def isolated_quarc_stream_id_counters():
    """Match ``quarc.orchestration.build._isolated_stream_ids`` so scratch modules do not consume global ids."""
    from quarc.dsl.streams import _incoming_ids, _outgoing_ids

    incoming_before = _incoming_ids.current
    outgoing_before = _outgoing_ids.current
    _incoming_ids.reset()
    _outgoing_ids.reset()
    try:
        yield
    finally:
        _incoming_ids.set_current(incoming_before)
        _outgoing_ids.set_current(outgoing_before)


def attach_pool_targets_from_quarc_module(
    module: Any,
    targets: Sequence[Union["ParameterTable", "Parameter"]],
) -> None:
    """Copy ``incoming_stream_spec`` / ``outgoing_stream_spec`` ids from Quarc ``module._structs`` onto pool targets."""
    structs = module._structs
    for obj in targets:
        name = default_quarc_struct_name(obj)
        if name not in structs:
            raise KeyError(
                f"Quarc module has no struct {name!r}; keys: {list(structs)!r}"
            )
        st = structs[name]
        inc = st.incoming_stream_spec.id if st.incoming_stream_spec is not None else None
        out = st.outgoing_stream_spec.id if st.outgoing_stream_spec is not None else None
        obj._attach_quarc_stream_specs(incoming_id=inc, outgoing_id=out)
