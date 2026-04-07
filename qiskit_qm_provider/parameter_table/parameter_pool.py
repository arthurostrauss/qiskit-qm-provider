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

"""Registry of OPNIC packet targets and stable integer ids for Quarc-aligned stream wiring.

The pool holds every registered :class:`~qiskit_qm_provider.parameter_table.ParameterTable`
and standalone :class:`~qiskit_qm_provider.parameter_table.Parameter` that received an id via
:meth:`ParameterPool.get_id`. Integer ids match registration order; Quarc-facing **edge** stream
ids (incoming/outgoing per struct) are assigned by :meth:`ParameterPool.prepare_opnic_quarc_hybrid_packets`
using the same ``add_struct`` ordering as an in-memory Quarc ``BaseModule``.
"""

from __future__ import annotations

import itertools
import weakref
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .parameter_table import ParameterTable
    from .parameter import Parameter


class ParameterPool:
    _counter = itertools.count(1)
    _registry: Dict[int, ParameterTable | Parameter] = {}
    # Standalone OPNIC Parameter instances until ``set_index`` (table column) or first ``stream_id`` read.
    _opnic_unbound: Dict[int, weakref.ref] = {}
    #: Optional replacement for the legacy ``opnic_wrapper`` module (e.g. Quarc-generated ``*_opnic``).
    _opnic_transport: Optional[Any] = None
    _opnic_runtime: Optional[Any] = None
    _opnic_runtime_packets: Dict[str, Any] = {}

    @classmethod
    def _opnic_unbound_cleanup(cls, oid: int) -> Callable[[weakref.ReferenceType], None]:
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
    def _materialize_opnic_unbound_stream_ids(cls) -> None:
        for oid in list(cls._opnic_unbound.keys()):
            ref = cls._opnic_unbound.get(oid)
            if ref is None:
                continue
            p = ref()
            if p is not None:
                _ = p.stream_id

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
    def rebind_parameter_table_id(cls, table: "ParameterTable", new_id: int) -> None:
        """
        Move an OPNIC :class:`~qiskit_qm_provider.parameter_table.ParameterTable` to a stable pool id (e.g. from
        ``rl_qoc_pool_table_id`` in ``rl_qoc_quarc_specs.json``) and refresh its Quarc packet type so
        :func:`~qiskit_qm_provider.parameter_table.quarc_naming.default_quarc_struct_name` matches codegen.

        Raises if ``new_id`` is already held by a different registered object.
        """
        from .input_type import InputType

        if table.input_type != InputType.OPNIC:
            raise ValueError("rebind_parameter_table_id is only valid for OPNIC ParameterTable instances.")
        old_id = int(table._id)
        new_id = int(new_id)
        if old_id == new_id:
            existing = cls._registry.get(new_id)
            if existing is not None and existing is not table:
                raise ValueError(
                    f"Pool id {new_id} is already registered to a different object ({type(existing).__name__})."
                )
            if existing is None:
                cls._assert_unique_registered_name(table)
                cls._registry[new_id] = table
            table._recompute_opnic_qua_packet_type()
            return
        occupant = cls._registry.get(new_id)
        if occupant is not None and occupant is not table:
            raise ValueError(
                f"Cannot rebind table {table.name!r} to id {new_id}: id already held by {type(occupant).__name__}."
            )
        if old_id in cls._registry and cls._registry[old_id] is table:
            del cls._registry[old_id]
        cls._assert_unique_registered_name(table)
        cls._registry[new_id] = table
        table._id = new_id
        for p in table.parameters:
            if p._stream_id is not None:
                raise ValueError(
                    f"Parameter {p.name!r} already has stream_id {p._stream_id}; "
                    f"rebind table ids only before QUA declare_variables assigns column stream ids."
                )
        table._recompute_opnic_qua_packet_type()

    @classmethod
    def rebind_standalone_opnic_parameter_stream_id(cls, param: "Parameter", new_id: int) -> None:
        """
        Register a standalone OPNIC :class:`~qiskit_qm_provider.parameter_table.Parameter` at ``new_id`` (e.g.
        ``rl_qoc_pool_stream_id`` in specs), replacing any prior registry entry for ``param``.
        """
        from .input_type import InputType
        from .parameter import Parameter

        if not isinstance(param, Parameter):
            raise TypeError(f"Expected Parameter, got {type(param)!r}")
        if param.input_type != InputType.OPNIC:
            raise ValueError("rebind_standalone_opnic_parameter_stream_id requires InputType.OPNIC.")
        if param._table_indices:
            raise ValueError("Use rebind_parameter_table_id for parameters that belong to a ParameterTable.")
        new_id = int(new_id)
        for oid, reg in list(cls._registry.items()):
            if reg is param:
                del cls._registry[oid]
                break
        cls._opnic_unbound.pop(id(param), None)
        cls._assert_unique_registered_name(param)
        param._stream_id = new_id
        cls._registry[new_id] = param

    @classmethod
    def get_id(cls, obj: Any = None) -> int:
        """
        Next monotonic id. With ``obj``, registers it after a name uniqueness check (when ``obj.name`` is set).
        """
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
        """Clear registry and counters only (does not mutate or reset registered objects)."""
        cls._counter = itertools.count(1)
        cls._registry.clear()
        cls._opnic_unbound.clear()
        cls._opnic_transport = None
        cls._opnic_runtime = None
        cls._opnic_runtime_packets = {}

    @classmethod
    def set_opnic_transport_module(cls, module: Any | None) -> None:
        """
        Use ``module`` for OPNIC ``push_to_opx`` / ``fetch_from_opx`` instead of importing ``opnic_wrapper``.

        When ``None`` (default), :meth:`get_opnic_transport_module` imports ``opnic_wrapper``.

        ``module`` must expose the same symbols the provider expects from ``opnic_wrapper``:

        - ``OutgoingPacket``, ``send_packet`` (outgoing packets / tables)
        - ``read_packet``, ``wait_for_packets`` (incoming)
        - optionally ``close_streams`` (called from :meth:`close_streams`)

        Quarc-generated OPNIC Python modules often use ``recv()`` / ``send()`` on per-struct attributes; expose a thin
        adapter object as ``module`` that implements the ``opnic_wrapper`` call signatures using your runtime's
        ``setup_opnic`` / struct handles and numeric ``stream_id``\\ s from
        :meth:`prepare_opnic_quarc_hybrid_packets`.
        """
        cls._opnic_transport = module
        cls._opnic_runtime = None
        cls._opnic_runtime_packets = {}

    @classmethod
    def get_opnic_transport_module(cls) -> Any:
        """Return the active OPNIC transport module (override or ``opnic_wrapper``)."""
        if cls._opnic_transport is not None:
            return cls._opnic_transport
        import opnic_wrapper as mod

        return mod

    @classmethod
    def get_opnic_runtime(cls) -> Any:
        """
        Return an initialized OPNIC runtime for Quarc-generated bindings.

        Supports both:
        - explicit runtime object set via ``set_opnic_transport_module(runtime)``
        - module exposing ``setup_opnic(...)`` (called once and cached)
        """
        if cls._opnic_runtime is not None:
            return cls._opnic_runtime
        mod = cls.get_opnic_transport_module()
        if hasattr(mod, "data_packet"):
            cls._opnic_runtime = mod
            cls._refresh_runtime_packet_registry(mod)
            return mod
        setup = getattr(mod, "setup_opnic", None)
        if callable(setup):
            try:
                runtime = setup(init_now=True)
            except TypeError:
                runtime = setup()
            cls._opnic_runtime = runtime
            cls._refresh_runtime_packet_registry(runtime)
            return runtime
        raise ImportError(
            "No OPNIC runtime found. Provide an adapter with send_packet/read_packet APIs, "
            "or a Quarc-style module/runtime exposing setup_opnic(...)->runtime.data_packet."
        )

    @classmethod
    def _refresh_runtime_packet_registry(cls, runtime: Any) -> None:
        packets: Dict[str, Any] = {}
        for attr_name in dir(runtime):
            if attr_name.startswith("_"):
                continue
            try:
                attr_value = getattr(runtime, attr_name)
            except Exception:
                continue
            if callable(getattr(attr_value, "send", None)) and callable(
                getattr(attr_value, "recv", None)
            ):
                packets[attr_name] = attr_value
        cls._opnic_runtime_packets = packets

    @classmethod
    def get_opnic_runtime_packet(
        cls, *, target_obj: Any | None = None, fallback_data_packet: bool = True
    ) -> Any:
        runtime = cls.get_opnic_runtime()
        if not cls._opnic_runtime_packets:
            cls._refresh_runtime_packet_registry(runtime)
        if target_obj is not None:
            try:
                from .quarc_naming import default_quarc_struct_name

                struct_name = default_quarc_struct_name(target_obj)
                if struct_name in cls._opnic_runtime_packets:
                    return cls._opnic_runtime_packets[struct_name]
            except Exception:
                pass
            target_name = getattr(target_obj, "name", None)
            if isinstance(target_name, str) and target_name in cls._opnic_runtime_packets:
                return cls._opnic_runtime_packets[target_name]
        if fallback_data_packet and "data_packet" in cls._opnic_runtime_packets:
            return cls._opnic_runtime_packets["data_packet"]
        if len(cls._opnic_runtime_packets) == 1:
            return next(iter(cls._opnic_runtime_packets.values()))
        raise ImportError(
            "Could not resolve OPNIC runtime packet handle for target. "
            "Expected runtime attribute named after struct/binding."
        )

    @classmethod
    def iter_opnic_quarc_packet_targets(cls) -> List[Union["ParameterTable", "Parameter"]]:
        """
        OPNIC :class:`~qiskit_qm_provider.parameter_table.ParameterTable` and standalone
        :class:`~qiskit_qm_provider.parameter_table.Parameter` instances, sorted like Quarc ``add_struct`` order.
        """
        from .input_type import InputType
        from .parameter import Parameter
        from .parameter_table import ParameterTable
        from .quarc_naming import quarc_packet_sort_key

        cls._materialize_opnic_unbound_stream_ids()
        tables = cls.iter_opnic_parameter_tables()
        standalones: List[Parameter] = []
        for obj in cls.get_all_objs():
            if (
                isinstance(obj, Parameter)
                and obj.input_type == InputType.OPNIC
                and not obj._table_indices
            ):
                standalones.append(obj)
        targets: List[Union[ParameterTable, Parameter]] = [*tables, *standalones]
        targets.sort(key=quarc_packet_sort_key)
        return targets

    @classmethod
    def prepare_opnic_quarc_hybrid_packets(
        cls,
        *,
        quarc_module_class_name: str | None = None,
    ) -> List[Union["ParameterTable", "Parameter"]]:
        """
        Build an in-memory Quarc ``BaseModule`` (``add_struct`` per target), then copy edge stream ids onto targets.

        Requires the ``quarc`` package. Uses isolated Quarc stream counters so this does not perturb other Quarc
        usage in-process. Call after the pool snapshot is final and before QUA uses ``declare_external_stream``.
        """
        targets = cls.iter_opnic_quarc_packet_targets()
        if not targets:
            return []
        try:
            from .quarc_live_module import (
                attach_pool_targets_from_quarc_module,
                build_quarc_base_module_from_specs,
                isolated_quarc_stream_id_counters,
                specs_from_quarc_packet_targets,
            )
        except ImportError as exc:
            raise ImportError(
                "ParameterPool.prepare_opnic_quarc_hybrid_packets requires the `quarc` package."
            ) from exc
        with isolated_quarc_stream_id_counters():
            specs = specs_from_quarc_packet_targets(
                targets,
                module_class_name=quarc_module_class_name,
            )
            module = build_quarc_base_module_from_specs(specs)
            attach_pool_targets_from_quarc_module(module, targets)
        return targets

    @classmethod
    def rebind_from_quarc_specs(cls, specs: Dict[str, Any]) -> None:
        """Rebind pool table / standalone stream ids using ``structs[].rl_qoc_binding`` in a specs dict."""
        from .parameter import Parameter
        from .parameter_table import ParameterTable
        from .quarc_naming import default_quarc_struct_name

        structs: List[Dict[str, Any]] = list(specs.get("structs") or [])
        by_binding: Dict[str, Dict[str, Any]] = {}
        for st in structs:
            b = st.get("rl_qoc_binding")
            if isinstance(b, str) and b:
                by_binding[b] = st

        for obj in cls.get_all_objs():
            if isinstance(obj, ParameterTable):
                spec = by_binding.get(obj.name)
                if spec is None:
                    continue
                pid = spec.get("rl_qoc_pool_table_id")
                if pid is None:
                    continue
                cls.rebind_parameter_table_id(obj, int(pid))
                expected = spec.get("struct_name")
                if isinstance(expected, str) and default_quarc_struct_name(obj) != expected:
                    raise ValueError(
                        f"After rebind, struct name mismatch for table {obj.name!r}: "
                        f"got {default_quarc_struct_name(obj)!r}, specs have {expected!r}"
                    )
            elif isinstance(obj, Parameter):
                if obj._table_indices:
                    continue
                spec = by_binding.get(obj.name)
                if spec is None:
                    continue
                sid = spec.get("rl_qoc_pool_stream_id")
                if sid is None:
                    continue
                cls.rebind_standalone_opnic_parameter_stream_id(obj, int(sid))
                expected = spec.get("struct_name")
                if isinstance(expected, str) and default_quarc_struct_name(obj) != expected:
                    raise ValueError(
                        f"After rebind, struct name mismatch for parameter {obj.name!r}: "
                        f"got {default_quarc_struct_name(obj)!r}, specs have {expected!r}"
                    )

    @classmethod
    def attach_opnic_streams_from_specs_dict(
        cls,
        specs: Dict[str, Any],
        *,
        module_class_name: str | None = None,
    ) -> List[Union["ParameterTable", "Parameter"]]:
        """
        Rebind from embedded codegen ids, build an in-memory Quarc module, and copy edge stream ids onto targets.

        Same contract as ``rl_qoc.qua.quarc.parameter_hydration.attach_streams_from_specs_dict`` (kept in rl_qoc
        for JSON path / manifest helpers).
        """
        data = dict(specs)
        if module_class_name is not None:
            data["module_class_name"] = module_class_name
        cls.rebind_from_quarc_specs(data)
        targets = cls.iter_opnic_quarc_packet_targets()
        if not targets:
            return []
        try:
            from .quarc_live_module import (
                attach_pool_targets_from_quarc_module,
                build_quarc_base_module_from_specs,
                isolated_quarc_stream_id_counters,
            )
        except ImportError as exc:
            raise ImportError(
                "ParameterPool.attach_opnic_streams_from_specs_dict requires the `quarc` package."
            ) from exc
        with isolated_quarc_stream_id_counters():
            module = build_quarc_base_module_from_specs(data)
            attach_pool_targets_from_quarc_module(module, targets)
        return targets

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
    def close_streams(cls) -> None:
        """Call ``close_streams`` on the active OPNIC transport module when available (no-op otherwise)."""
        try:
            mod = cls.get_opnic_transport_module()
        except ImportError:
            return
        closer = getattr(mod, "close_streams", None)
        if callable(closer):
            closer()
