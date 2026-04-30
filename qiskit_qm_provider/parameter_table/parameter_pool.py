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

"""Registry of OPNIC packet targets, stable integer ids, and the Quarc module accumulator.

The pool serves two intertwined purposes:

1. **Stable integer ids and named registry.**
   Every :class:`ParameterTable` that is constructed registers itself in the pool with a
   monotonically increasing integer id, used for ``stream_id`` assignment in the legacy /
   non-Quarc paths and as a stable key for cross-references. Names are unique across the
   registry (and across the pending standalone OPNIC parameter list — see below).

2. **Quarc module accumulator (single slot).**
   At most one :class:`quarc.BaseModule` instance is bound to the pool at any time, in
   ``_quarc_module``. Two pipelines populate it:

   * **Pipeline 1 — module-first.** ``ParameterPool.from_quarc_module(my_module)`` wraps
     each pre-declared struct in ``my_module`` as a :class:`ParameterTable` (binding the
     existing :class:`QuaStructHandle` to ``table._var``), sets the pool slot to
     ``my_module``, and sweeps any *previously-pending* OPNIC ``ParameterTable``\\ s — i.e.
     tables the user already constructed before calling ``from_quarc_module`` — onto
     ``my_module`` via ``my_module.add_struct``.

   * **Pipeline 2 — parameters-first.** The user declares ``Parameter``/``ParameterTable``
     objects with ``input_type=InputType.OPNIC`` and *later* calls
     :meth:`ParameterPool.to_quarc_module`. That call lazily creates a default
     :class:`quarc.BaseModule` (if no module is bound yet), sweeps every unemitted OPNIC
     table in the registry through ``module.add_struct``, and returns the populated
     module.

   Once a module is bound (whichever pipeline got there first), every *new* OPNIC
   ``ParameterTable`` created afterwards eagerly emits its struct onto that bound module
   at construction time — there is no second sweep needed. The two pipelines are mutually
   exclusive: calling :meth:`from_quarc_module` after the slot has been bound (e.g. by an
   earlier :meth:`to_quarc_module` call) raises, and so does
   :meth:`set_quarc_module` on a bound slot. Use :meth:`reset` to start over.

   Quarc's ``module.add_struct`` is **append-only** — once a struct/handle has been
   registered (and its stream ids consumed from the global incoming/outgoing counters),
   there is no remove or replace. All emission paths in this module respect that.

**Standalone OPNIC parameters.** A :class:`Parameter` constructed with
``input_type=InputType.OPNIC`` is *not* immediately added to the main registry; it is
appended to ``_pending_standalone_opnic``. As long as it stays there it can still be
attached to a regular ``ParameterTable`` via :meth:`Parameter.set_index` (which removes
it from the pending list). If the user calls :meth:`Parameter.declare_variable` (or any
other transport-level method) on a parameter that is still pending, the parameter is
*promoted*: a synthetic single-field :class:`ParameterTable` is constructed wrapping it,
the table emits its struct, and the parameter becomes locked. After this point the
parameter cannot be attached to any other ``ParameterTable``.

:meth:`ParameterPool.to_quarc_module` does **not** automatically promote pending
standalone parameters — it only sweeps OPNIC tables that already exist in the registry.
This is intentional: ``to_quarc_module`` should be a side-effect-free getter / lazy
builder that doesn't make standalone-vs-table decisions on the user's behalf. Pending
parameters that the user never declares are simply not in the resulting module.

**Cross-process note.** The ``BaseModule`` returned by :meth:`to_quarc_module` is a real
in-process Quarc module. If the QUA-side and classical-side code execute in the same
Python session (typical for ``quarc.run()`` driven from a single entry point) they share
the same module instance. For multi-process deployments the classical side must rebuild
the wrapper tables — typically by calling :meth:`from_quarc_module` against the same
``BaseModule`` subclass / serialized representation.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .parameter_table import ParameterTable
    from .parameter import Parameter


class ParameterPool:
    _counter = itertools.count(1)
    _registry: Dict[int, ParameterTable | Parameter] = {}
    #: Solo OPNIC :class:`Parameter` instances that have *not yet* been attached to a
    #: ``ParameterTable`` and have not been promoted to a synthetic standalone table.
    #: Populated at ``Parameter.__init__`` (for OPNIC only); cleared by
    #: ``Parameter.set_index`` when the parameter joins a non-synthetic table, or by
    #: standalone promotion at first :meth:`Parameter.declare_variable`.
    _pending_standalone_opnic: List["Parameter"] = []
    #: The single Quarc :class:`BaseModule` slot owned by the pool.
    #: Set by :meth:`from_quarc_module` (Pipeline 1) or lazily by
    #: :meth:`to_quarc_module` (Pipeline 2). Re-binding a different module without an
    #: intervening :meth:`reset` raises.
    _quarc_module: Optional[Any] = None

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------

    @classmethod
    def _assert_unique_registered_name(cls, obj: Any) -> None:
        """Reject a registration that would shadow an existing name.

        Names are unique across the *visible* universe: the main registry plus the
        pending standalone OPNIC parameter list (since a pending parameter logically
        owns its name and may later be promoted to a synthetic table named after it).
        """
        name = getattr(obj, "name", None)
        if not isinstance(name, str) or not name:
            return
        for existing in cls._registry.values():
            if existing is not obj and getattr(existing, "name", None) == name:
                raise ValueError(
                    f"Duplicate pool registration name {name!r}; "
                    f"already held by {type(existing).__name__}."
                )
        for pending in cls._pending_standalone_opnic:
            if pending is not obj and pending.name == name:
                raise ValueError(
                    f"Duplicate pool registration name {name!r}; "
                    f"already held by a pending standalone OPNIC Parameter."
                )

    @classmethod
    def _lookup_parameter_by_name(cls, name: str) -> Optional["Parameter"]:
        """Return the canonical Parameter named ``name``, or ``None`` if there is none.

        Walks (in order):
        - parameters inside any registered :class:`ParameterTable`,
        - direct ``Parameter`` instances stored in the registry (rare — usually only
          appears for synthetic-standalone parameters where the table is the registered
          owner),
        - pending standalone OPNIC parameters.
        """
        from .parameter import Parameter
        from .parameter_table import ParameterTable

        for obj in cls._registry.values():
            if isinstance(obj, ParameterTable):
                for param in obj.parameters:
                    if param.name == name:
                        return param
            elif isinstance(obj, Parameter):
                if obj.name == name:
                    return obj
        for param in cls._pending_standalone_opnic:
            if param.name == name:
                return param
        return None

    # ------------------------------------------------------------------------
    # Id allocation / registry CRUD
    # ------------------------------------------------------------------------

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
        """Clear all pool state — registry, id counter, pending list, and Quarc module."""
        cls._counter = itertools.count(1)
        cls._registry.clear()
        cls._pending_standalone_opnic.clear()
        cls._quarc_module = None

    @classmethod
    def get_all_ids(cls) -> List[int]:
        return list(cls._registry.keys())

    @classmethod
    def get_all_objs(cls) -> List[ParameterTable | Parameter]:
        return list(cls._registry.values())

    @classmethod
    def get_all(cls) -> Dict[int, ParameterTable | Parameter]:
        return dict(cls._registry)

    @classmethod
    def iter_opnic_parameter_tables(cls) -> List["ParameterTable"]:
        """Return all registered OPNIC :class:`ParameterTable`\\ s, sorted by id.

        Includes synthetic single-field tables produced by standalone OPNIC parameter
        promotion.
        """
        from .input_type import InputType
        from .parameter_table import ParameterTable

        tables: List[ParameterTable] = []
        for obj in cls.get_all_objs():
            if isinstance(obj, ParameterTable) and obj.input_type == InputType.OPNIC:
                tables.append(obj)
        tables.sort(key=lambda t: t._id)
        return tables

    @classmethod
    def iter_standalone_opnic_parameters(cls) -> List["Parameter"]:
        """Return every OPNIC Parameter that is, or was, treated as standalone.

        The returned list is the *union* of:
        - parameters still in :data:`_pending_standalone_opnic` (created with
          ``input_type=OPNIC`` and never attached to a non-synthetic table), and
        - parameters that have been promoted into a synthetic single-field
          :class:`ParameterTable` (i.e. the table has ``_is_synthetic_standalone == True``).

        Order: pending parameters first (in insertion order), then promoted ones in
        registry order.
        """
        from .parameter_table import ParameterTable

        result: List["Parameter"] = list(cls._pending_standalone_opnic)
        for obj in cls.get_all_objs():
            if (
                isinstance(obj, ParameterTable)
                and getattr(obj, "_is_synthetic_standalone", False)
                and obj.parameters
            ):
                result.append(obj.parameters[0])
        return result

    # ------------------------------------------------------------------------
    # Pending standalone OPNIC parameter list
    # ------------------------------------------------------------------------

    @classmethod
    def _register_pending_standalone_opnic(cls, parameter: "Parameter") -> None:
        """Internal: append a freshly-constructed OPNIC Parameter to the pending list.

        Called from :meth:`Parameter.__init__`. Idempotent — if the parameter is already
        present it is not duplicated.
        """
        if parameter in cls._pending_standalone_opnic:
            return
        cls._pending_standalone_opnic.append(parameter)

    @classmethod
    def _unregister_pending_standalone_opnic(cls, parameter: "Parameter") -> None:
        """Internal: remove a Parameter from the pending list when it is attached to a
        non-synthetic ParameterTable, or when it is promoted to a synthetic table.

        Idempotent — quietly does nothing if the parameter isn't in the list.
        """
        try:
            cls._pending_standalone_opnic.remove(parameter)
        except ValueError:
            pass

    # ------------------------------------------------------------------------
    # Quarc module slot
    # ------------------------------------------------------------------------

    @classmethod
    def has_quarc_module(cls) -> bool:
        """``True`` iff a Quarc module is currently bound to the pool."""
        return cls._quarc_module is not None

    @classmethod
    def set_quarc_module(cls, module: Any) -> None:
        """Bind an externally-built ``BaseModule`` (or subclass) instance.

        Raises :class:`RuntimeError` if a module is already bound. Use :meth:`reset`
        first if you genuinely need to swap.
        """
        if cls._quarc_module is not None:
            raise RuntimeError(
                "ParameterPool already has a Quarc module bound. "
                "Call ParameterPool.reset() before binding a different one."
            )
        cls._quarc_module = module

    @classmethod
    def quarc_module(cls) -> Any:
        """Return the bound Quarc module, lazily creating a default ``BaseModule()`` if
        none has been bound yet.

        After this method is called once with no pre-bound module, ``_quarc_module`` is
        set to a default :class:`quarc.BaseModule` instance and that becomes the slot;
        subsequent OPNIC ``ParameterTable`` constructions emit eagerly into it.
        """
        if cls._quarc_module is None:
            try:
                from quarc import BaseModule
            except ImportError as exc:
                raise ImportError(
                    "ParameterPool.quarc_module requires the `quarc` package."
                ) from exc
            cls._quarc_module = BaseModule()
        return cls._quarc_module

    # ------------------------------------------------------------------------
    # Pipeline 1 — module-first
    # ------------------------------------------------------------------------

    @classmethod
    def from_quarc_module(cls, module: Any) -> Dict[str, "ParameterTable"]:
        """Wrap every struct in ``module`` as a :class:`ParameterTable` (Pipeline 1).

        For each ``StructSpec`` already registered in ``module._structs`` the matching
        :class:`QuaStructHandle` is bound to the produced ``ParameterTable._var``
        (no fresh ``add_struct`` call is made for those — they are already in ``module``).

        The provided ``module`` then becomes the pool's bound accumulator:
        - any OPNIC :class:`ParameterTable` that was *previously* constructed (and is
          therefore still pending — i.e. ``_is_emitted == False``) is now swept onto
          ``module`` via ``module.add_struct``, with a fresh :class:`QuaStructHandle`
          bound to its ``_var``;
        - any OPNIC :class:`ParameterTable` constructed *afterwards* will eagerly emit
          onto ``module`` at construction time (no need to call ``from_quarc_module``
          again);
        - any pending standalone OPNIC :class:`Parameter` is left untouched until the
          user actually declares it (per the standalone promotion rules).

        Raises :class:`RuntimeError` if a module is already bound — call :meth:`reset`
        first if you really want to rebind.

        Returns a ``dict`` mapping the snake-case struct name to the wrapper
        :class:`ParameterTable`, for convenience.
        """
        if cls._quarc_module is not None:
            raise RuntimeError(
                "ParameterPool already has a Quarc module bound — cannot call "
                "from_quarc_module() again. Call ParameterPool.reset() first if you "
                "really want to rebind."
            )

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
                    f"Struct {struct_name!r} has no stream specs; cannot build "
                    f"ParameterTable."
                )

            annotations = get_type_hints(struct_spec.struct)
            params: List[Any] = []
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
                        f"Struct {struct_name!r} field {field_name!r} has unsupported "
                        f"annotation {annotation!r}; expected Scalar[T] or Array[T, N]."
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

            table = ParameterTable(params, name=struct_name, _quarc_handle=handle)
            table._direction = direction
            table._input_type = InputType.OPNIC
            result[pascal_to_snake_case(struct_name)] = table

        # Bind the slot *before* sweeping, so the sweep's `add_struct` calls happen on
        # the user-provided module (not on a freshly-allocated default one).
        cls._quarc_module = module

        # Sweep any previously-pending OPNIC tables onto this module. Wrappers built
        # above already had ``_quarc_handle=`` so they are already emitted.
        cls._sweep_pending_opnic_tables_into(module)

        return result

    # ------------------------------------------------------------------------
    # Pipeline 2 — parameters-first
    # ------------------------------------------------------------------------

    @classmethod
    def to_quarc_module(cls, module: Optional[Any] = None) -> Any:
        """Return the pool-owned ``BaseModule`` accumulator, building it on demand.

        Side-effect-free w.r.t. *standalone* OPNIC parameters: pending solo Parameters
        are **not** promoted by this call — they only get promoted on
        :meth:`Parameter.declare_variable` (or another transport-level call). This
        matches the "promote on use" semantics: the user is the one to decide they
        intended a parameter to be standalone, by actually using it.

        Behaviour:

        * If ``module`` is provided, behaves like :meth:`set_quarc_module` (raises if
          a module is already bound) followed by the same sweep below.
        * If no module is bound yet, lazily creates a default ``quarc.BaseModule()``
          and binds it to the pool.
        * Sweeps every OPNIC :class:`ParameterTable` in the registry that has not yet
          been emitted (``_is_emitted == False``) and emits it via ``module.add_struct``.
          Each emission consumes one or two stream ids from Quarc's global counters.
        * Idempotent on subsequent calls — the second invocation finds nothing pending
          and just returns the same module reference.

        Returns the bound ``BaseModule``.
        """
        if module is not None:
            cls.set_quarc_module(module)
        target = cls.quarc_module()
        cls._sweep_pending_opnic_tables_into(target)
        return target

    @classmethod
    def _sweep_pending_opnic_tables_into(cls, module: Any) -> None:
        """Emit every unemitted OPNIC ParameterTable in the registry onto ``module``.

        Used by both :meth:`from_quarc_module` (after it sets the slot to a user-supplied
        module) and :meth:`to_quarc_module` (after it sets the slot to a default-built
        module). Skips tables that already have ``_is_emitted == True`` (e.g. wrappers
        constructed with a pre-existing ``_quarc_handle``).
        """
        from .input_type import InputType
        from .parameter_table import ParameterTable

        for obj in list(cls._registry.values()):
            if (
                isinstance(obj, ParameterTable)
                and obj.input_type == InputType.OPNIC
                and not obj._is_emitted
            ):
                obj._emit_to_module(module)
