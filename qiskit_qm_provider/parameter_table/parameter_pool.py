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
     each pre-declared struct in ``my_module`` as a :class:`ParameterTable`, *or* — for
     structs with exactly **one field** — as a standalone :class:`Parameter` (the wrapper
     table is still created and marked ``_is_synthetic_standalone=True`` so all OPNIC
     transport delegates through it correctly; see *1-field struct promotion* below).
     The pool sets the slot to ``my_module``. If ``my_module`` is a
     :class:`~qiskit_qm_provider.QiskitQMModule`, its ``__init__`` is responsible for
     sweeping any previously-pending OPNIC ``ParameterTable``\\ s and pending standalone
     OPNIC ``Parameter``\\ s onto itself (see
     :meth:`QiskitQMModule._sweep_preexisting_opnic`).

   * **Pipeline 2 — parameters-first.** The user declares ``Parameter``/``ParameterTable``
     objects with ``input_type=InputType.OPNIC`` and *later* either creates a
     :class:`~qiskit_qm_provider.QiskitQMModule` (whose ``__init__`` binds the slot and
     sweeps the registry / pending-list) or calls
     :meth:`ParameterPool.to_quarc_module`. The latter binds (or lazily creates) a
     :class:`~qiskit_qm_provider.QiskitQMModule`, whose ``__init__`` performs the same
     sweep as an explicit ``QiskitQMModule()`` construction.

   Once a module is bound (whichever pipeline got there first), every *new* OPNIC
   ``ParameterTable`` created afterwards eagerly emits its struct onto that bound module
   at construction time. The two pipelines are mutually exclusive: calling
   :meth:`from_quarc_module` after the slot has been bound (e.g. by an earlier
   :meth:`to_quarc_module` call) raises, and so does :meth:`set_quarc_module` on a bound
   slot. Use :meth:`reset` to start over.

   Quarc's ``module.add_struct`` is **append-only** — once a struct/handle has been
   registered (and its stream ids consumed from the global incoming/outgoing counters),
   there is no remove or replace. All emission paths in this module respect that.

**Standalone OPNIC parameters.** A :class:`Parameter` constructed with
``input_type=InputType.OPNIC`` is *not* immediately added to the main registry; it is
appended to ``_pending_standalone_opnic``. As long as it stays there it can still be
attached to a regular ``ParameterTable`` via :meth:`Parameter.set_index` (which removes
it from the pending list). If the user calls :meth:`Parameter.declare` (or any
other transport-level method) on a parameter that is still pending, the parameter is
*promoted*: a synthetic single-field :class:`ParameterTable` is constructed wrapping it,
the table emits its struct at that ``declare()``, and the parameter becomes locked.
After this point the parameter cannot be attached to any other ``ParameterTable``.

:meth:`ParameterPool.to_quarc_module` does **not** automatically promote pending
standalone parameters — it only binds the module slot. Promotion of pre-existing
pending standalone OPNIC parameters at module-binding time is performed by
:class:`QiskitQMModule` via :meth:`QiskitQMModule._sweep_preexisting_opnic` so that
their structs are part of the deployment artifact even before the QUA program runs.

**Cross-process note.** The ``BaseModule`` returned by :meth:`to_quarc_module` is a real
in-process Quarc module. If the QUA-side and classical-side code execute in the same
Python session (typical for ``quarc.run()`` driven from a single entry point) they share
the same module instance. For multi-process deployments the classical side must rebuild
the wrapper tables — typically by calling :meth:`from_quarc_module` against the same
``BaseModule`` subclass / serialized representation.

**Measurement outputs (separate namespace).** Compiled-circuit measurement tables and
their :class:`~qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField`
fields (including loose-bit ``_bitN`` keys) are tracked in private pool containers
(``_measurement_outcome_tables``, ``_measurement_register_fields``), not in
``_registry``. They share no ids with runtime/OPNIC objects. Field names may match
runtime struct fields or creg names; resolve handles via ``comp.outputs`` vs your
input ``ParameterTable``. :meth:`reset` clears measurement registries together with
the runtime registry.
"""

from __future__ import annotations

import itertools
import warnings
import weakref
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .parameter_table import ParameterTable
    from .parameter import Parameter


def _resolve_lookup_key(name: str, attr_name: Optional[str] = None) -> str:
    """Resolve the snake_case lookup key for a serialized parameter or struct name.

    Shared by OPNIC runtime endpoint resolution (``_structs`` keys are PascalCase while
    generated ``Runtime`` objects expose snake_case attributes) and non-OPNIC
    ``parameter_specs`` reconstruction (``INPUT_STREAM``, ``IO1``, ``IO2``, compile-time).
    """
    if attr_name:
        return attr_name
    try:
        from quarc.naming import pascal_to_snake_case

        return pascal_to_snake_case(name)
    except ImportError:
        return name


def _assign_reconstructed(result: Dict[str, Any], key: str, value: Any, *, source_name: Any = None) -> None:
    """Insert ``value`` under ``key`` in a reconstruction result dict, warning on collision.

    Two distinct serialized names can collapse to the same snake_case lookup key; without a
    warning the earlier object would be silently overwritten. Shared by
    :meth:`ParameterPool._from_quarc_module_dict` and
    :meth:`QiskitQMModule.reconstruct_non_opnic` so both paths behave identically.
    """
    if key in result:
        suffix = f" (from {source_name!r})" if source_name is not None else ""
        warnings.warn(
            f"Reconstruction key {key!r} collides{suffix}; the earlier object is being "
            "overwritten. Two serialized names map to the same snake_case key.",
            stacklevel=2,
        )
    result[key] = value


def _resolve_runtime_endpoint(runtime: Any, struct_name: str) -> Any:
    """Resolve a live OPNIC runtime endpoint for a ``_structs`` key.

    Tries snake_case first (generated ``Runtime`` packet names), then the raw key for
    structs that were already snake_case at emission time (e.g. ``input_state_vars``).
    """
    snake_name = _resolve_lookup_key(struct_name)
    for key in (snake_name, struct_name):
        endpoint = getattr(runtime, key, None)
        if endpoint is not None:
            return endpoint
    return None


class ParameterPool:
    """Process-global registry of OPNIC packet targets and the bound Quarc module.

    .. note::
       **Single-thread constraint.** :class:`ParameterPool` stores all of its state
       on class-level attributes. It is designed for single-threaded use — one Quarc
       session per process at a time. There is no per-thread isolation; running
       multiple QUA programs concurrently from the same process is not supported.
       Per-thread isolation via :mod:`threading.local` is a planned future
       enhancement.
    """

    _counter = itertools.count(1)
    #: id -> ParameterTable. The weak view drives lookups; ``_strong_refs`` (below) pins the
    #: same objects for the lifetime of the current pool/program so a table whose only other
    #: reference is a caller's local cannot be garbage-collected mid-program and silently
    #: vanish from iteration (which would mis-classify its parameters as standalone). Both
    #: are cleared by :meth:`reset`, so the lifetime stays process-/program-scoped — the
    #: single-module-per-process invariant means there is nothing to leak across resets.
    #: Non-weak-referenceable objects passed to :meth:`get_id` (e.g. bare sentinels) are not
    #: stored. Parameters are not registered directly — they are reached through their table.
    _registry: "weakref.WeakValueDictionary[int, ParameterTable]" = weakref.WeakValueDictionary()
    #: Strong-reference companion to ``_registry`` (id -> registered object), keeping
    #: registered tables alive until :meth:`reset`. See ``_registry`` note above.
    _strong_refs: Dict[int, Any] = {}
    #: Solo OPNIC :class:`Parameter` instances that have *not yet* been attached to a
    #: ``ParameterTable`` and have not been promoted to a synthetic standalone table.
    #: Populated at ``Parameter.__init__`` (for OPNIC only); cleared by
    #: ``Parameter.set_index`` when the parameter joins a non-synthetic table, or by
    #: standalone promotion at first :meth:`Parameter.declare`.
    _pending_standalone_opnic: List["Parameter"] = []
    #: The single Quarc :class:`BaseModule` slot owned by the pool.
    #: Set by :meth:`from_quarc_module` (Pipeline 1) or lazily by
    #: :meth:`to_quarc_module` (Pipeline 2). Re-binding a different module without an
    #: intervening :meth:`reset` raises.
    _quarc_module: Optional[Any] = None
    #: Program-scope token under which OPNIC structs were first emitted/declared. The
    #: sync-hook / classical flow is valid for exactly one QUA program, so declaring OPNIC
    #: tables under a second *distinct* program scope raises (see
    #: :meth:`guard_single_program_opnic`). Cleared by :meth:`reset`.
    _committed_scope_token: Optional[Any] = None
    #: :class:`~qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable`
    #: instances tracked separately from the runtime/OPNIC registry (not assigned ids).
    _measurement_outcome_tables: weakref.WeakSet[Any] = weakref.WeakSet()
    #: :class:`~qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField`
    #: handles (including loose-bit ``_bitN`` fields), weakref-tracked. A ``WeakSet`` gives
    #: O(1) identity dedup and auto-prunes dead entries (no manual sweep).
    _measurement_register_fields: weakref.WeakSet[Any] = weakref.WeakSet()

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------

    @classmethod
    def _register_measurement_outcome_table(cls, table: Any) -> None:
        """Track a compiled-circuit measurement table outside the runtime registry."""
        cls._measurement_outcome_tables.add(table)
        for field in table.parameters:
            cls._register_measurement_register_field(field)

    @classmethod
    def _register_measurement_register_field(cls, field: Any) -> None:
        """Track a measurement register handle (creg or loose bit) outside the runtime registry."""
        cls._measurement_register_fields.add(field)

    @classmethod
    def _reset_measurement_registries(cls) -> None:
        """Clear measurement-only registries (called from :meth:`reset`)."""
        cls._measurement_outcome_tables = weakref.WeakSet()
        cls._measurement_register_fields = weakref.WeakSet()

    @classmethod
    def iter_measurement_outcome_tables(cls) -> Iterator[Any]:
        """Yield live :class:`~qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable` instances (debug/introspection)."""
        yield from cls._measurement_outcome_tables

    @classmethod
    def iter_measurement_register_fields(cls) -> Iterator[Any]:
        """Yield live :class:`~qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField` handles (debug/introspection)."""
        yield from cls._measurement_register_fields

    @classmethod
    def _assert_unique_registered_name(cls, obj: Any) -> None:
        """Reject a registered ``ParameterTable`` whose name shadows another table.

        Table names must be unique because they become Quarc struct keys
        (``module._structs[struct.__name__]`` would silently overwrite a collision). This
        is *table*-name uniqueness only — it is **not** the old global ``Parameter``
        name-dedup (which was removed): two parameters may share a name across tables.
        """
        name = getattr(obj, "name", None)
        if not isinstance(name, str) or not name:
            return
        for existing in cls._registry.values():
            if existing is not obj and getattr(existing, "name", None) == name:
                raise ValueError(
                    f"Duplicate pool registration name {name!r}; " f"already held by {type(existing).__name__}."
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

    @classmethod
    def lookup_runtime_parameter(cls, name: str) -> Optional["Parameter"]:
        """Return the runtime :class:`Parameter` named ``name`` (excludes measurement fields)."""
        return cls._lookup_parameter_by_name(name)

    # ------------------------------------------------------------------------
    # Id allocation / registry CRUD
    # ------------------------------------------------------------------------

    @classmethod
    def get_id(cls, obj: Any = None) -> int:
        # Validate uniqueness BEFORE consuming a counter id, so a rejected registration
        # does not leave a gap in the (otherwise contiguous, deterministic) id sequence.
        if obj is not None:
            cls._assert_unique_registered_name(obj)
        next_id = next(cls._counter)
        if obj is not None:
            try:
                cls._registry[next_id] = obj
                cls._strong_refs[next_id] = obj
            except TypeError:
                # Object is not weak-referenceable (e.g. a bare sentinel used only to
                # bump the counter). The id is still allocated; it just isn't stored.
                pass
        return next_id

    @classmethod
    def get_obj(cls, id: int) -> ParameterTable | Parameter:
        return cls._registry[id]

    @classmethod
    def reset(cls) -> None:
        """Clear all pool state — runtime registry, measurement registries, and Quarc module."""
        cls._counter = itertools.count(1)
        cls._registry.clear()
        cls._strong_refs.clear()
        cls._pending_standalone_opnic.clear()
        cls._quarc_module = None
        cls._committed_scope_token = None
        cls._reset_quarc_stream_id_counters()
        cls._reset_measurement_registries()

    @staticmethod
    def _reset_quarc_stream_id_counters() -> None:
        """Reset Quarc's process-global incoming/outgoing stream-id counters.

        These live in ``quarc.dsl.streams`` and are NOT scope-bound, so without this a
        second module built in the same process would mint ids continuing from the first
        (drifting from any stored/firmware ids). Best-effort: silently skips if Quarc is
        not installed or its internals move.
        """
        try:
            from quarc.dsl.streams import _incoming_ids, _outgoing_ids

            _incoming_ids.reset()
            _outgoing_ids.reset()
        except Exception:
            pass

    @classmethod
    def guard_single_program_opnic(cls, token: Any, *, table_name: str = "<unknown>") -> None:
        """Enforce that OPNIC structs are declared under a single QUA program scope.

        The first OPNIC ``declare()`` records its program-scope ``token``; a later
        ``declare()`` under a *different* token raises. Declaring multiple OPNIC tables in
        the same program is fine (same token). Call :meth:`reset` to start a new program.
        """
        if token is None:
            return
        if cls._committed_scope_token is None:
            cls._committed_scope_token = token
        elif cls._committed_scope_token is not token:
            raise RuntimeError(
                f"OPNIC ParameterTable {table_name!r} is being declared in a second, "
                f"distinct QUA program scope. The OPNIC / sync-hook flow supports exactly "
                f"one program per process. Call ParameterPool.reset() before building a "
                f"new program if this is intentional."
            )

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

        Union of parameters still in :data:`_pending_standalone_opnic` and parameters
        promoted into a synthetic single-field :class:`ParameterTable`
        (``_is_synthetic_standalone == True``). Pending entries come first, then
        promoted entries in registry order.
        """
        from .parameter_table import ParameterTable

        result: List["Parameter"] = list(cls._pending_standalone_opnic)
        for obj in cls.get_all_objs():
            if isinstance(obj, ParameterTable) and getattr(obj, "_is_synthetic_standalone", False) and obj.parameters:
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
        """Return the bound Quarc module, lazily creating a default
        :class:`~qiskit_qm_provider.QiskitQMModule` if none has been bound yet.

        After this method is called once with no pre-bound module, ``_quarc_module`` is
        set to a :class:`~qiskit_qm_provider.QiskitQMModule` instance (which sweeps
        pre-existing OPNIC pool state during ``__init__``) and that becomes the slot;
        subsequent OPNIC ``ParameterTable`` constructions emit eagerly into it.
        """
        if cls._quarc_module is None:
            try:
                from ..qiskit_qm_module import QiskitQMModule
            except ImportError as exc:
                raise ImportError("ParameterPool.quarc_module requires the `quarc` package.") from exc
            cls._quarc_module = QiskitQMModule()
        return cls._quarc_module

    # ------------------------------------------------------------------------
    # Pipeline 1 — module-first
    # ------------------------------------------------------------------------

    @classmethod
    def from_quarc_module(
        cls,
        module: Any,
        opnic_runtime: Optional[Any] = None,
    ) -> Dict[str, "ParameterTable | Parameter"]:
        """Wrap every struct in ``module`` and return a parameter dict.

        This method operates in **two modes** depending on whether ``module`` is a live
        :class:`quarc.BaseModule` instance or a plain ``dict`` (e.g. loaded from a
        ``rl_qoc_state.json`` file).

        ---

        **Mode 1 — Module object (quantum / generation side)**

        Input is a :class:`quarc.BaseModule` (or subclass) instance.  ``_struct_handles``
        are already populated (built by :meth:`add_struct` during construction).  Each
        handle is paired with its struct spec to create a ``ParameterTable`` with
        ``_quarc_handle=handle``.

        If ``module`` is also a ``QiskitQMModule`` (has ``parameter_specs``), non-OPNIC
        objects are reconstructed via :meth:`~QiskitQMModule.reconstruct_non_opnic` and
        merged into the result dict.

        The provided ``module`` becomes the pool's bound accumulator.  Any previously-
        pending OPNIC :class:`ParameterTable`\\s are swept onto it.

        **Mode 2 — Dictionary (classical entrypoint side)**

        Input is a ``dict`` (the JSON loaded directly from the state file).  The
        ``opnic_runtime`` argument **must** be provided.

        * ``"_structs"`` key → for each struct, the live endpoint is resolved on
          ``opnic_runtime`` via :func:`_resolve_runtime_endpoint` (snake_case attribute
          first, then the raw ``_structs`` key).
        * ``"parameter_specs"`` key (optional) → non-OPNIC objects reconstructed via
          :meth:`~.parameter_table.ParameterTable.from_spec` /
          :meth:`~.parameter.Parameter.from_spec`.

        ---

        **1-field struct promotion rule (both modes):** structs with exactly one field are
        returned as a standalone :class:`Parameter`; the wrapper table is flagged as
        ``_is_synthetic_standalone=True`` so all OPNIC transport delegates through it.

        Args:
            module: A :class:`quarc.BaseModule` instance *or* a state dict.
            opnic_runtime: Required when ``module`` is a dict.  Must expose each struct
                from ``"_structs"`` as a snake_case runtime attribute (with a fallback
                to the raw ``_structs`` key for already-snake struct names).

        Returns:
            A ``dict`` mapping snake-case struct name to the wrapper object
            (:class:`ParameterTable` for multi-field structs, :class:`Parameter` for
            1-field structs), plus any non-OPNIC entries keyed by ``attr_name`` / name.

        Raises:
            RuntimeError: If a module is already bound (Mode 1).
            ValueError: If ``module`` is a dict but ``opnic_runtime`` is not provided,
                or if ``"_structs"`` key is absent.
        """
        if isinstance(module, dict):
            return cls._from_quarc_module_dict(module, opnic_runtime=opnic_runtime)
        return cls._from_quarc_module_object(module)

    @classmethod
    def _from_quarc_module_object(cls, module: Any) -> Dict[str, "ParameterTable | Parameter"]:
        """Internal: Mode 1 — live module object."""
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

            from ..qiskit_qm_module import QiskitQMModule
        except ImportError as exc:
            raise ImportError("ParameterPool.from_quarc_module requires the `quarc` package.") from exc

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

        result: Dict[str, ParameterTable | Parameter] = {}
        for (struct_name, struct_spec), handle in zip(structs.items(), handles):
            has_incoming = struct_spec.incoming_stream_spec is not None
            has_outgoing = struct_spec.outgoing_stream_spec is not None
            if has_incoming and has_outgoing:
                direction = Direction.BOTH
            elif has_incoming:
                # Quarc incoming spec = data flows into the QUA program = qiskit INCOMING.
                direction = Direction.INCOMING
            elif has_outgoing:
                direction = Direction.OUTGOING
            else:
                raise ValueError(f"Struct {struct_name!r} has no stream specs; cannot build " f"ParameterTable.")

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

            table = ParameterTable(
                params,
                name=struct_name,
                _quarc_handle=handle,
                _var_is_quarc_handle=True,
            )
            snake_name = pascal_to_snake_case(struct_name)
            if len(params) == 1:
                # Promote 1-field structs to a standalone Parameter. Mark the wrapper
                # table as synthetic-standalone so that Parameter.is_stand_alone is True
                # and OPNIC transport delegates back through this table's handle.
                table._is_synthetic_standalone = True
                result[snake_name] = params[0]
            else:
                result[snake_name] = table

        # Bind the slot. Pre-existing OPNIC tables and pending standalone OPNIC
        # Parameters are swept by ``QiskitQMModule.__init__`` (see
        # :meth:`QiskitQMModule._sweep_preexisting_opnic`); plain ``BaseModule``
        # instances coming through this path do not have any pre-existing pool state
        # to sweep (the registry is owned by qiskit-qm-provider, not by Quarc).
        cls._quarc_module = module

        # Non-OPNIC reconstruction (QiskitQMModule and subclasses only)
        if isinstance(module, QiskitQMModule) and module.parameter_specs:
            result.update(module.reconstruct_non_opnic())

        return result

    @classmethod
    def _from_quarc_module_dict(
        cls,
        state: Dict[str, Any],
        opnic_runtime: Optional[Any],
    ) -> Dict[str, "ParameterTable | Parameter"]:
        """Internal: Mode 2 — plain state dict + live runtime."""
        if opnic_runtime is None:
            raise ValueError(
                "opnic_runtime is required when module is a dict. "
                "Pass the live Quarc runtime object so struct endpoints can be resolved."
            )
        if "_structs" not in state:
            raise ValueError(
                "State dict must contain a '_structs' key. "
                "Ensure the state was produced by QiskitQMModule.to_dict() or "
                "RLQoCModule.to_dict()."
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
            raise ImportError("ParameterPool.from_quarc_module requires the `quarc` package.") from exc

        atomic_to_qua_type: Dict[type, Any] = {float: fixed, int: int, bool: bool}

        def _default_value(qua_type: Any, length: Optional[int]) -> Any:
            scalar = False if qua_type is bool else (0 if qua_type is int else 0.0)
            if length is None:
                return scalar
            return [scalar] * length

        result: Dict[str, ParameterTable | Parameter] = {}

        # --- OPNIC structs from _structs key + runtime endpoints ---
        for struct_name, struct_info in state["_structs"].items():
            endpoint = _resolve_runtime_endpoint(opnic_runtime, struct_name)
            if endpoint is None:
                lookup_key = _resolve_lookup_key(struct_name)
                warnings.warn(
                    f"opnic_runtime has no attribute {lookup_key!r} (or {struct_name!r}); "
                    f"skipping struct reconstruction.",
                    stacklevel=3,
                )
                continue

            fields_raw: Dict[str, Any] = struct_info.get("struct", {})
            has_in = struct_info.get("incoming_stream_spec") is not None
            has_out = struct_info.get("outgoing_stream_spec") is not None
            if has_in and has_out:
                direction = Direction.BOTH
            elif has_in:
                # Quarc incoming spec = data flows into the QUA program = qiskit INCOMING.
                direction = Direction.INCOMING
            elif has_out:
                direction = Direction.OUTGOING
            else:
                warnings.warn(
                    f"Struct {struct_name!r} has no stream specs in the state dict; " f"skipping.",
                    stacklevel=3,
                )
                continue

            params: List[Any] = []
            for field_name, fspec in fields_raw.items():
                atomic_str = fspec.get("type", "float")
                length_val = fspec.get("length", 1)
                if atomic_str == "float":
                    qua_type = fixed
                elif atomic_str == "int":
                    qua_type = int
                else:
                    qua_type = bool
                value = _default_value(qua_type, int(length_val) if int(length_val) > 1 else None)
                params.append(
                    Parameter(
                        field_name,
                        value=value,
                        qua_type=qua_type,
                        input_type=InputType.OPNIC,
                        direction=direction,
                    )
                )

            table = ParameterTable(
                params,
                name=struct_name,
                _quarc_handle=endpoint,
                _var_is_quarc_handle=False,
            )
            table._sync_stream_ids_from_state(struct_info)
            snake_name = _resolve_lookup_key(struct_name)
            if len(params) == 1:
                table._is_synthetic_standalone = True
                _assign_reconstructed(result, snake_name, params[0], source_name=struct_name)
            else:
                _assign_reconstructed(result, snake_name, table, source_name=struct_name)

        # --- Non-OPNIC params from parameter_specs key (optional) ---
        for spec in state.get("parameter_specs", []):
            key = _resolve_lookup_key(spec["name"], spec.get("attr_name"))
            if spec.get("is_table", False) and "fields" in spec:
                _assign_reconstructed(result, key, ParameterTable.from_spec(spec), source_name=spec["name"])
            else:
                _assign_reconstructed(result, key, Parameter.from_spec(spec), source_name=spec["name"])

        return result

    # ------------------------------------------------------------------------
    # Pipeline 2 — parameters-first
    # ------------------------------------------------------------------------

    @classmethod
    def to_quarc_module(cls, module: Optional[Any] = None) -> Any:
        """Bind the pool slot to ``module`` (or lazily create a default
        :class:`~qiskit_qm_provider.QiskitQMModule`) and return it.

        Behaviour:

        * If ``module`` is provided, behaves like :meth:`set_quarc_module` — binds the
          slot, raising :class:`RuntimeError` if a different module is already bound.
          Sweeping pre-existing pool state is only automatic when ``module`` is a
          :class:`~qiskit_qm_provider.QiskitQMModule` (via
          :meth:`QiskitQMModule._sweep_preexisting_opnic` in ``__init__``).
        * If no module is bound yet, lazily creates a default
          :class:`~qiskit_qm_provider.QiskitQMModule` (same sweep semantics as an
          explicit ``QiskitQMModule()`` construction).
        * Idempotent on subsequent calls — returns the same module reference.

        Returns the bound module (typically :class:`~qiskit_qm_provider.QiskitQMModule`).
        """
        if module is not None:
            cls.set_quarc_module(module)
        return cls.quarc_module()
