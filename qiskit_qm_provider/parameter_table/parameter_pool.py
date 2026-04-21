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

"""Registry of OPNIC packet targets and stable integer ids.

Stream wiring for Quarc-backed training flows through :meth:`ParameterPool.from_quarc_module`, which binds each
struct to a :class:`~qiskit_qm_provider.parameter_table.ParameterTable` whose ``_var`` is a
``QuaStructHandle`` (or pybind endpoint) with ``send`` / ``recv``.
"""

from __future__ import annotations

import itertools
import weakref
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .parameter_table import ParameterTable
    from .parameter import Parameter


class ParameterPool:
    _counter = itertools.count(1)
    _registry: Dict[int, ParameterTable | Parameter] = {}
    _opnic_unbound: Dict[int, weakref.ref] = {}

    @classmethod
    def _opnic_unbound_cleanup(
        cls, oid: int
    ) -> Callable[[weakref.ReferenceType], None]:
        def _cb(_ref: weakref.ReferenceType) -> None:
            cls._opnic_unbound.pop(oid, None)

        return _cb

    @classmethod
    def _track_opnic_parameter_pending_table(cls, param: "Parameter") -> None:
        from .input_type import InputType

        if param.input_type != InputType.OPNIC:
            return
        oid = id(param)
        cls._opnic_unbound[oid] = weakref.ref(param, cls._opnic_unbound_cleanup(oid))

    @classmethod
    def _release_opnic_unbound_parameter(cls, param: "Parameter") -> None:
        cls._opnic_unbound.pop(id(param), None)

    @classmethod
    def _assert_unique_registered_name(cls, obj: Any) -> None:
        name = getattr(obj, "name", None)
        if not isinstance(name, str) or not name:
            return
        for existing in cls._registry.values():
            if existing is not obj and getattr(existing, "name", None) == name:
                raise ValueError(
                    f"Duplicate pool registration name {name!r}; "
                    f"already held by {type(existing).__name__}."
                )

    @classmethod
    def get_id(cls, obj: Any = None) -> int:
        next_id = next(cls._counter)
        if obj is not None:
            cls._assert_unique_registered_name(obj)
            cls._registry[next_id] = obj
        return next_id

    @classmethod
    def get_obj(cls, id: int) -> ParameterTable | Parameter:
        return cls._registry[id]

    @classmethod
    def reset(cls) -> None:
        cls._counter = itertools.count(1)
        cls._registry.clear()
        cls._opnic_unbound.clear()

    @classmethod
    def get_all_ids(cls) -> List[int]:
        return list(cls._registry.keys())

    @classmethod
    def get_all_objs(cls) -> List[ParameterTable | Parameter]:
        return list(cls._registry.values())

    @classmethod
    def iter_opnic_parameter_tables(cls) -> List["ParameterTable"]:
        from .input_type import InputType
        from .parameter_table import ParameterTable

        tables: List[ParameterTable] = []
        for obj in cls.get_all_objs():
            if isinstance(obj, ParameterTable) and obj.input_type == InputType.OPNIC:
                tables.append(obj)
        tables.sort(key=lambda t: t._id)
        return tables

    @classmethod
    def get_all(cls) -> Dict[int, ParameterTable | Parameter]:
        return dict(cls._registry)

    @classmethod
    def from_quarc_module(cls, module: Any) -> Dict[str, "ParameterTable"]:
        """Build ``ParameterTable`` instances from a ``quarc.BaseModule``'s registered structs.

        See module docstring in the previous revision; behavior unchanged.
        """
        from typing import get_args, get_origin

        from typing_extensions import get_type_hints

        from qm.qua import fixed

        from .input_type import Direction, InputType
        from .parameter import Parameter
        from .parameter_table import ParameterTable

        try:
            from quarc import Array as QuarcArray, Scalar as QuarcScalar
            from quarc.naming import pascal_to_snake_case
        except ImportError as exc:
            raise ImportError(
                "ParameterPool.from_quarc_module requires the `quarc` package."
            ) from exc

        atomic_to_qua_type: Dict[type, Any] = {float: fixed, int: int, bool: bool}

        def _default_value(qua_type: Any, length: Optional[int]) -> Any:
            scalar = False if qua_type is bool else (0 if qua_type is int else 0.0)
            if length is None:
                return scalar
            return [scalar] * length

        structs = module._structs
        handles = module._struct_handles
        if len(structs) != len(handles):
            raise RuntimeError(
                "Module has inconsistent struct bookkeeping: "
                f"{len(structs)} struct specs vs {len(handles)} struct handles."
            )

        result: Dict[str, ParameterTable] = {}
        for (struct_name, struct_spec), handle in zip(structs.items(), handles):
            has_incoming = struct_spec.incoming_stream_spec is not None
            has_outgoing = struct_spec.outgoing_stream_spec is not None
            if has_incoming and has_outgoing:
                direction = Direction.BOTH
            elif has_incoming:
                direction = Direction.OUTGOING
            elif has_outgoing:
                direction = Direction.INCOMING
            else:
                raise ValueError(
                    f"Struct {struct_name!r} has no stream specs; cannot build ParameterTable."
                )

            annotations = get_type_hints(struct_spec.struct)
            params: List[Parameter] = []
            for field_name, annotation in annotations.items():
                origin = get_origin(annotation)
                args = get_args(annotation)
                if origin is QuarcScalar:
                    (atomic,) = args
                    length: Optional[int] = None
                elif origin is QuarcArray:
                    atomic, array_length = args
                    length = int(array_length)
                else:
                    raise TypeError(
                        f"Struct {struct_name!r} field {field_name!r} has unsupported annotation "
                        f"{annotation!r}; expected Scalar[T] or Array[T, N]."
                    )
                qua_type = atomic_to_qua_type[atomic]
                value = _default_value(qua_type, length)
                params.append(
                    Parameter(
                        field_name,
                        value=value,
                        qua_type=qua_type,
                        input_type=InputType.OPNIC,
                        direction=direction,
                    )
                )

            table = ParameterTable(params, name=struct_name)
            table._var = handle
            table._direction = direction
            table._input_type = InputType.OPNIC
            result[pascal_to_snake_case(struct_name)] = table

        return result
