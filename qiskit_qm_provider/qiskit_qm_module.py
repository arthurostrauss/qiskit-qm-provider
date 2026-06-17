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

"""QiskitQMModule: a quarc.BaseModule extension with a unified parameter API.

Author: Arthur Strauss
Date: 2026-05-05
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Tuple

from pydantic import Field
from quarc import Array, BaseModule
from quarc import Direction as QuarcDirection
from quarc import Scalar, Struct

from qiskit_qm_provider.parameter_table.parameter_pool import ParameterPool

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _field_annotation(atomic: str, length: int) -> Any:
    """Build a quarc ``Scalar`` / ``Array`` annotation from an atomic type name and length.

    Args:
        atomic: One of ``"float"``, ``"int"``, or ``"bool"``.
        length: Field size.  ``1`` → ``Scalar[T]``; ``> 1`` → ``Array[T, length]``.
    """
    _types: Dict[str, type] = {"float": float, "int": int, "bool": bool}
    t = _types.get(atomic, float)
    return Array[t, length] if length > 1 else Scalar[t]


# ---------------------------------------------------------------------------
# QiskitQMModule
# ---------------------------------------------------------------------------


class QiskitQMModule(BaseModule):
    """A :class:`quarc.BaseModule` extension that adds a unified parameter API.

    ``QiskitQMModule`` sits between :class:`quarc.BaseModule` and domain-specific
    modules such as ``RLQoCModule``. It provides:

    * A ``parameter_specs`` Pydantic field — a list of serialisable spec dicts for
      **non-OPNIC** parameters (``INPUT_STREAM``, ``IO1``, ``IO2``, plain QUA vars).
      OPNIC parameters are serialised through the inherited ``_structs`` key via
      :meth:`quarc.BaseModule.to_dict`.

    * Automatic registration with :class:`ParameterPool` at construction time. The
      module binds itself as the pool's bound module slot, then sweeps any
      pre-existing OPNIC ``ParameterTable``\\ s (and pending standalone OPNIC
      ``Parameter``\\ s) onto itself via :meth:`_sweep_preexisting_opnic`. Any
      pre-existing non-OPNIC tables / standalone parameters are captured into
      :attr:`parameter_specs` via :meth:`_sweep_preexisting_non_opnic`. Subsequent
      OPNIC ``ParameterTable``\\ s emit eagerly at construction (driven by
      ``ParameterTable.__init__``) and subsequent non-OPNIC objects register at
      ``declare()`` time. There is no ``add_parameter`` method — registration is
      automatic.

    * :meth:`reconstruct_non_opnic` — rebuilds non-OPNIC objects from
      ``parameter_specs`` using the ``from_spec()`` factories on ``Parameter`` /
      ``ParameterTable``.

    * :meth:`iter_all_params` — unified ``(kind, key, handle_or_spec)`` iterator.

    Only one ``QiskitQMModule`` (or subclass) instance can be alive in a process at
    a time, mirroring the single-slot model of :class:`ParameterPool`. Constructing
    a second instance without an intervening :meth:`ParameterPool.reset` raises
    :class:`RuntimeError`.

    On construction, any ``_structs`` key present in *data* (e.g. when re-creating
    from a JSON state dict) is replayed via :meth:`_replay_from_structs_data` so
    Quarc struct handles are populated without requiring a live runtime.
    """

    parameter_specs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Serialisable spec dicts for non-OPNIC parameters/tables.",
    )

    # ------------------------------------------------------------------
    # QMM connection metadata — serialised into the module JSON so the
    # classical entrypoint can reconnect after quarc's deploy phase and
    # discover the running job via qmm.get_jobs() / list_open_qms().
    # ------------------------------------------------------------------
    qmm_host: Optional[str] = Field(default=None, description="QMM server host.")
    qmm_port: Optional[int] = Field(default=None, description="QMM server port.")

    def __init__(self, **data: Any) -> None:
        # Pop private-attr keys that Pydantic can't consume as fields.
        structs_data: Dict[str, Any] = data.pop("_structs", None) or {}
        data.pop("_struct_handles", None)
        super().__init__(**data)

        # Bind this module to the ParameterPool so all subsequent OPNIC
        # ParameterTable / Parameter constructions are routed onto self. Mirrors
        # the single-slot constraint enforced by ParameterPool.
        if ParameterPool.has_quarc_module():
            raise RuntimeError(
                "ParameterPool already has a Quarc module bound. "
                "Call ParameterPool.reset() before creating a new QiskitQMModule."
            )
        ParameterPool.to_quarc_module(module=self)

        # Sweep pre-existing pool state onto self so the contract-first invariant
        # holds: every OPNIC parameter declared before this module exists must be
        # part of the module's struct list, and every non-OPNIC parameter must be
        # captured in parameter_specs.
        self._sweep_preexisting_opnic()
        self._sweep_preexisting_non_opnic()

        if structs_data:
            self._replay_from_structs_data(structs_data)

    # ------------------------------------------------------------------
    # Struct replay (safe without a live runtime)
    # ------------------------------------------------------------------

    def _replay_from_structs_data(self, structs_data: Dict[str, Any]) -> None:
        """Replay ``_structs`` JSON data by calling :meth:`add_struct` for each entry.

        This is safe to call without a live Quarc runtime — ``add_struct`` only
        allocates in-memory stream-id counters and builds ``QuaStructHandle`` objects.

        Args:
            structs_data: The value of the ``"_structs"`` key from a :meth:`to_dict`
                serialisation (a dict of ``struct_name → StructSpecSchema`` dicts).
        """
        for struct_name, struct_info in structs_data.items():
            fields: Dict[str, Any] = struct_info.get("struct", {})
            if not fields:
                continue
            has_in = struct_info.get("incoming_stream_spec") is not None
            has_out = struct_info.get("outgoing_stream_spec") is not None
            if has_in and has_out:
                quarc_dir = QuarcDirection.BOTH
            elif has_in:
                quarc_dir = QuarcDirection.INCOMING
            elif has_out:
                quarc_dir = QuarcDirection.OUTGOING
            else:
                continue  # no stream specs — skip
            annotations: Dict[str, Any] = {
                fn: _field_annotation(
                    str(fspec.get("type", "float")),
                    int(fspec.get("length", 1)),
                )
                for fn, fspec in fields.items()
            }
            struct_cls = Struct(struct_name=struct_name, **annotations)
            self.add_struct(struct_cls, quarc_dir)

    # ------------------------------------------------------------------
    # Pre-existing pool state sweeps (run during __init__)
    # ------------------------------------------------------------------

    def _sweep_preexisting_opnic(self) -> None:
        """Emit every pre-existing OPNIC :class:`ParameterTable` and pending
        standalone OPNIC :class:`Parameter` onto this module.

        Two paths:

        - **Path 1 — registry sweep.** Every unemitted OPNIC ``ParameterTable``
          currently in :attr:`ParameterPool._registry` is emitted via
          :meth:`ParameterTable._emit_to_module`.
        - **Path 2 — pending standalone promotion.** Every solo OPNIC
          ``Parameter`` in :attr:`ParameterPool._pending_standalone_opnic` is
          promoted to a synthetic single-field table via
          :meth:`Parameter._promote_to_synthetic_standalone_table`. The synthetic
          table emits eagerly because the pool slot is already bound to ``self``.

        Path 2 is required so that pre-existing standalone OPNIC parameters land
        in the deployment artifact even if the QUA program has not yet executed
        a transport call on them. Without it, ``quarc.init_module`` /
        ``quarc.build`` would serialise an incomplete struct list.
        """
        from qiskit_qm_provider.parameter_table.input_type import InputType
        from qiskit_qm_provider.parameter_table.parameter_table import ParameterTable

        for obj in list(ParameterPool.get_all_objs()):
            if isinstance(obj, ParameterTable) and obj.input_type == InputType.OPNIC and not obj._is_emitted:
                obj._emit_to_module(self)

        for param in list(ParameterPool._pending_standalone_opnic):
            param._promote_to_synthetic_standalone_table()

    def _sweep_preexisting_non_opnic(self) -> None:
        """Capture every pre-existing non-OPNIC ``ParameterTable`` /
        standalone ``Parameter`` into :attr:`parameter_specs`.

        Idempotent by parameter / table name.
        """
        from qiskit_qm_provider.parameter_table.input_type import InputType
        from qiskit_qm_provider.parameter_table.parameter import Parameter
        from qiskit_qm_provider.parameter_table.parameter_table import ParameterTable

        existing_names = {s.get("name") for s in self.parameter_specs}
        for obj in list(ParameterPool.get_all_objs()):
            if isinstance(obj, ParameterTable) and obj.input_type != InputType.OPNIC:
                if obj.name not in existing_names:
                    self.parameter_specs.append(obj.to_spec())
                    existing_names.add(obj.name)
            elif isinstance(obj, Parameter) and obj.input_type != InputType.OPNIC and not obj.tables:
                if obj.name not in existing_names:
                    self.parameter_specs.append(obj.to_spec())
                    existing_names.add(obj.name)

    # ------------------------------------------------------------------
    # Reconstruction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spec_key(spec: Dict[str, Any]) -> str:
        """Resolve the lookup key for a non-OPNIC spec.

        Falls back to ``attr_name`` for backward compatibility with state files
        produced before the ``attr_name`` write side was removed; otherwise uses
        ``pascal_to_snake_case(spec['name'])`` so PascalCase parameter names map
        cleanly onto snake_case Python attributes (e.g. on
        ``QMEnvironment.circuit_params``).
        """
        try:
            from quarc.naming import pascal_to_snake_case
        except ImportError:
            pascal_to_snake_case = None  # type: ignore[assignment]

        attr_name = spec.get("attr_name")
        if attr_name:
            return attr_name
        name = spec["name"]
        if pascal_to_snake_case is None:
            return name
        return pascal_to_snake_case(name)

    def reconstruct_non_opnic(self) -> Dict[str, Any]:
        """Rebuild non-OPNIC ``Parameter`` / ``ParameterTable`` objects from
        :attr:`parameter_specs`.

        Returns:
            A dict mapping ``attr_name`` (legacy) or
            ``pascal_to_snake_case(name)`` (current) to the reconstructed object.
        """
        from qiskit_qm_provider.parameter_table.parameter import Parameter
        from qiskit_qm_provider.parameter_table.parameter_table import ParameterTable

        result: Dict[str, Any] = {}
        for spec in self.parameter_specs:
            key = self._spec_key(spec)
            if spec.get("is_table", False) and "fields" in spec:
                result[key] = ParameterTable.from_spec(spec)
            else:
                result[key] = Parameter.from_spec(spec)
        return result

    # ------------------------------------------------------------------
    # QMM / QM context binding and job retrieval
    # ------------------------------------------------------------------

    def bind_connection(
        self,
        qmm: Any,
    ) -> None:
        """Store QMM connection details so they survive JSON serialisation.

        Called automatically from :attr:`QMBackend.qm` on first open. In pure-quarc
        flows (no ``QMBackend``) call this manually before ``quarc.run()``:

        .. code-block:: python

            module.bind_connection(qmm)
            quarc.run(module=module, ...)

        Args:
            qmm: The :class:`qm.QuantumMachinesManager` (or cloud equivalent) whose
                 ``host`` and ``port`` are stored for later job discovery.
        """
        self.qmm_host = getattr(qmm, "host", None)
        self.qmm_port = getattr(qmm, "port", None)

    def get_running_job(
        self,
        *,
        timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> Any:
        """Return the currently running QM job — works in all execution modes.

        Resolution order:

        1. **IQCC sync-hook mode** — detected when ``-j`` / ``--jobId`` is present in
           ``sys.argv``.  The job is reconstructed from ``-j/-q/-i/-p`` arguments,
           mirroring ``iqcc_cloud_client.runtime.get_qm_job()``.

        2. **Local / QMBackend mode** — reconnects to the QMM using the stored
           :attr:`qmm_host` / :attr:`qmm_port` (bound automatically when
           ``QMBackend.qm`` is first accessed) and polls for a running job.
           On QOP 3.x: ``qmm.get_jobs(status=["Running"])``.
           Fallback (QOP 2.x): iterates ``qmm.list_open_qms()`` and probes each
           machine with ``qm.get_running_job()``.

        Args:
            timeout: Seconds to wait for a running job before raising
                :class:`TimeoutError` (local mode only, default 30 s).
            poll_interval: Polling cadence in seconds (local mode only, default 0.1 s).

        Raises:
            RuntimeError: If neither IQCC args nor stored QMM details are available.
            TimeoutError: If no running job appears within *timeout* seconds.
        """
        from .runtime import (
            _is_sync_hook_mode,
            _job_from_sync_hook_args,
            _poll_running_job_from_qmm,
        )

        if _is_sync_hook_mode():
            return _job_from_sync_hook_args()

        if not self.qmm_host:
            raise RuntimeError(
                "No QMM connection details available. Either:\n"
                "  • Use the QMBackend flow (QMBackend.qm auto-binds QMM details), or\n"
                "  • Call module.bind_connection(qmm) before quarc.run(), or\n"
                "  • Run in IQCC sync-hook mode (pass -j/-q/-i/-p as CLI args)."
            )

        import logging

        from qm import QuantumMachinesManager

        qmm = QuantumMachinesManager(host=self.qmm_host, port=self.qmm_port, log_level=logging.ERROR)
        return _poll_running_job_from_qmm(qmm, timeout=timeout, poll_interval=poll_interval)

    # ------------------------------------------------------------------
    # Unified iterator
    # ------------------------------------------------------------------

    def iter_all_params(self) -> Generator[Tuple[str, str, Any], None, None]:
        """Iterate over all registered parameters.

        Yields:
            ``(kind, key, handle_or_spec)`` tuples where:

            * ``kind`` is ``"opnic"`` or ``"non_opnic"``.
            * ``key`` is the struct name (OPNIC) or
              ``attr_name`` / ``pascal_to_snake_case(name)`` (non-OPNIC).
            * ``handle_or_spec`` is the ``QuaStructHandle`` (OPNIC) or spec dict
              (non-OPNIC).
        """
        for (name, _), handle in zip(self._structs.items(), self._struct_handles):
            yield "opnic", name, handle
        for spec in self.parameter_specs:
            yield "non_opnic", self._spec_key(spec), spec
