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

"""Per-PUB execution plans shared by the QM primitives.

A plan owns everything one PUB needs across the job: its input tables, the QUA macros that
declare and load those inputs inside the program, and the host-side push order (used both
for local pushes and for the IQCC sync hook, so the two cannot drift apart).

* :class:`PrimitivePlan` -- circuit parameters (see :mod:`.pub_parameters`).
* :class:`SamplerPlan` -- adds shots.
* :class:`EstimatorPlan` -- adds measurement grouping, observable codes and their packed
  delivery (one int row per binding), and the metadata used to assemble results.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from qiskit.quantum_info import PauliList
from qm.qua import declare, for_

from ..backend.backend_utils import get_non_trivial_observables, observables_support
from ..parameter_table import Direction, InputType, Parameter, ParameterTable, QUA2DArray
from .pub_parameters import (
    assign_obs_codes,
    assign_qiskit_row,
    build_pub_tables,
    check_fixed_range,
    delivery_rows,
    load_pub_row,
    obs_words,
    pack_obs_rows,
    push_obs_row,
    rcv_obs_row,
)
from .sync_hook import _qua_type_str, serialize_table

if TYPE_CHECKING:
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from qiskit.primitives.containers.sampler_pub import SamplerPub
    from ..primitives.qm_estimator import QMEstimatorOptions

HookSpec = Tuple[str, str, str]
"""``(input_type, name, qua_type)`` of one sync-hook push target."""


def _hook_specs(target) -> List[HookSpec]:
    """Sync-hook specs of a push target: one per field of a table, one for a :class:`Parameter`."""
    if isinstance(target, ParameterTable):
        spec = serialize_table(target)
        return [(spec["input_type"], p["name"], p["qua_type"]) for p in spec["params"]]
    return [(target.input_type.value, target.name, _qua_type_str(target))]


@dataclass(eq=False)
class PrimitivePlan:
    """Inputs of one PUB: the circuit table (what the circuit reads) and the delivery table.

    ``param_table`` is the delivery table; it is the circuit table itself for compile-time
    sweeps and OPNIC, and a dense row table for streamed inputs (see :mod:`.pub_parameters`).
    """

    pub: Any
    index: int
    circuit_table: Optional[ParameterTable]
    param_table: Optional[ParameterTable]

    def __post_init__(self):
        check_fixed_range(self.pub, self.index)
        self._param_values_qua: Optional[QUA2DArray] = None

    @property
    def num_bindings(self) -> int:
        return self.pub.parameter_values.size

    @property
    def streams_params(self) -> bool:
        return self.param_table is not None and self.param_table.input_type is not None

    @property
    def uses_compile_time_params(self) -> bool:
        return self.param_table is not None and bool(self.param_table.parameters) and self.param_table.input_type is None

    @property
    def streams_inputs(self) -> bool:
        """``True`` when the client must push anything for this PUB (decides the IQCC sync hook)."""
        return self.streams_params

    # ---------------------------------------------------------------------------------- QUA
    def declare_params(self) -> None:
        """QUA macro: declare the parameter tables, and the compile-time binding rows."""
        if self.circuit_table is None:
            return
        self.circuit_table.declare()
        if self.param_table is not self.circuit_table:
            self.param_table.declare()
        if self.uses_compile_time_params:
            self._param_values_qua = QUA2DArray(
                f"param_values_{self.param_table.name}",
                self.pub.parameter_values.ravel().as_array(self.pub.circuit.parameters),
            )
            self._param_values_qua.declare()

    def load_params(self, p) -> None:
        """QUA macro: make binding ``p`` (a QUA int) visible to the circuit table."""
        if self.uses_compile_time_params:
            assign_qiskit_row(self.pub.circuit, self.circuit_table, self._param_values_qua[p])
        elif self.streams_params:
            load_pub_row(self.pub.circuit, self.circuit_table, self.param_table)

    # --------------------------------------------------------------------------------- host
    def _pushes(self) -> Iterator[Tuple[Any, Any]]:
        """``(target, value)`` in push order: a delivery table with its per-binding values,
        or a single :class:`Parameter` with its value. Consumed by :meth:`push` and
        :meth:`sync_hook_pushes`."""
        if not self.streams_params:
            return
        for p, values in enumerate(delivery_rows(self.pub, self.circuit_table, self.param_table)):
            yield self.param_table, values
            yield from self._binding_extras(p)

    def _binding_extras(self, p: int) -> Iterator[Tuple[Any, Any]]:
        """Pushes that follow binding ``p``'s parameters (none for the sampler)."""
        return iter(())

    def push(self, qm_job, qm=None) -> None:
        """Push this PUB's streamed inputs to a running job."""
        for target, value in self._pushes():
            self._push_one(target, value, qm_job, qm)

    def _push_one(self, target, value, qm_job, qm) -> None:
        if isinstance(target, ParameterTable):
            target.push_to_opx(dict(zip(target.table, value)), qm_job, qm)
        else:
            target.push_to_opx(value, qm_job, qm)

    def sync_hook_pushes(self) -> Iterator[Tuple[HookSpec, Any]]:
        """``(spec, value)`` per push, in order, for the IQCC sync hook (specs built once per target)."""
        specs: Dict[int, List[HookSpec]] = {}
        for target, value in self._pushes():
            if id(target) not in specs:
                specs[id(target)] = _hook_specs(target)
            if isinstance(target, ParameterTable):
                yield from zip(specs[id(target)], value)
            else:
                yield specs[id(target)][0], self._hook_value(target, value)

    def _hook_value(self, target: Parameter, value):
        return value


@dataclass(eq=False)
class SamplerPlan(PrimitivePlan):
    """Plan for one :class:`~qiskit.primitives.containers.sampler_pub.SamplerPub`."""

    @classmethod
    def from_pub(cls, pub: "SamplerPub", input_type: Optional[InputType], index: int) -> "SamplerPlan":
        circuit_table, param_table = build_pub_tables(pub.circuit, input_type, f"param_table_{index}")
        return cls(pub, index, circuit_table, param_table)

    @property
    def shots(self) -> int:
        return self.pub.shots


def _basis_code(z: bool, x: bool) -> int:
    """Per-qubit measurement basis code consumed by the estimator's switch: 0 Z/I, 1 X, 2 Y."""
    return 0 if not x else (2 if z else 1)


def _estimator_readout(circuit) -> List[int]:
    """Check the estimator readout contract and return the measured qubits in ``__c`` order.

    :meth:`~qiskit_qm_provider.primitives.QMEstimatorV2.validate_estimator_pubs` prepares every
    PUB circuit this way: a ``__c`` register whose bit ``j`` measures qubit ``measured[j]``,
    preceded by a basis switch on that qubit driven by the input variable ``obs_j``. The
    plan writes the per-group basis codes into ``obs_j`` and reads the outcomes from ``__c``.
    """
    hint = "Prepare estimator PUBs with QMEstimatorV2.validate_estimator_pubs."
    creg = next((creg for creg in circuit.cregs if creg.name == "__c"), None)
    if creg is None:
        raise ValueError(f"Estimator circuit {circuit.name!r} has no '__c' readout register. {hint}")
    measured = [None] * len(creg)
    for instruction in circuit.data:
        if instruction.operation.name == "measure" and instruction.clbits[0] in creg:
            measured[creg.index(instruction.clbits[0])] = circuit.find_bit(instruction.qubits[0]).index
    if any(q is None for q in measured):
        raise ValueError(f"Estimator circuit {circuit.name!r}: '__c' is not fully measured. {hint}")
    missing = [f"obs_{j}" for j in range(len(measured)) if not circuit.has_var(f"obs_{j}")]
    if missing:
        raise ValueError(f"Estimator circuit {circuit.name!r} lacks the basis inputs {missing}. {hint}")
    return measured


def _group_observables(labels: Tuple[str, ...], measured: List[int], abelian_grouping: bool):
    """Measurement groups of one binding's Pauli terms.

    Returns the per-group basis codes (in ``__c`` order) and, per group, the
    ``(meas_paulis, orig_paulis)`` pair used to post-process its counts. ``labels`` must act
    trivially outside ``measured``, so restriction to ``measured`` is one-to-one.
    """
    if not labels:
        return [], []
    restricted = get_non_trivial_observables(PauliList(list(labels)), measured)
    original = dict(zip(restricted.to_labels(), labels))
    if abelian_grouping:
        groups = restricted.group_commuting(qubit_wise=True)
    else:
        groups = [restricted[[i]] for i in range(len(restricted))]
    codes, paulis = [], []
    for group in groups:
        z, x = np.logical_or.reduce(group.z), np.logical_or.reduce(group.x)
        codes.append(tuple(_basis_code(z[j], x[j]) for j in range(len(measured))))
        paulis.append((group, PauliList([original[label] for label in group.to_labels()])))
    return codes, paulis


@dataclass(eq=False)
class EstimatorPlan(PrimitivePlan):
    """Plan for one :class:`~qiskit.primitives.containers.estimator_pub.EstimatorPub`.

    The PUB's circuit must follow the estimator readout contract (see :func:`_estimator_readout`).
    Grouping is done per distinct set of Pauli terms (shared by all bindings that request the
    same terms), on the terms restricted to the measured qubits (``__c[j]`` measures
    ``active_qubits[j]``, rotated by ``obs_j``). ``metadata``, the observable codes and the
    ``__c`` stream share one order: binding-major, groups in order.
    """

    metadata: List[dict] = field(default_factory=list)
    param_indices: Optional[np.ndarray] = None
    active_qubits: List[Any] = field(default_factory=list)
    obs_codes: List[List[Tuple[int, ...]]] = field(default_factory=list)
    obs_table: Optional[ParameterTable] = None
    obs_row: Optional[Parameter] = None
    obs_rows: List[List[int]] = field(default_factory=list)
    words_per_obs: int = 1

    def __post_init__(self):
        super().__post_init__()
        self._obs_rows_qua: Optional[QUA2DArray] = None
        self._obs_word = self._obs_idx = self._obs_counter = None

    @classmethod
    def from_pub(
        cls, pub: "EstimatorPub", options: "QMEstimatorOptions", plan_index: Optional[int] = None
    ) -> "EstimatorPlan":
        """Plan ``pub``; ``plan_index`` (its position in the job) names its tables uniquely."""
        circuit = pub.circuit
        index = plan_index if plan_index is not None else 0
        tag = plan_index if plan_index is not None else circuit.name
        if pub.parameter_values.size == 0:
            raise ValueError(
                f"PUB {index} has an empty bindings array (shape {pub.parameter_values.shape}); "
                "provide at least one set of parameter values."
            )
        if options.input_type == InputType.OPNIC:
            raise NotImplementedError(
                "Estimator observables over OPNIC are not supported yet; use INPUT_STREAM, IO1/IO2 "
                "or compile-time (input_type=None)."
            )
        measured = _estimator_readout(circuit)
        unmeasured = sorted(set(np.flatnonzero(observables_support(pub.observables)).tolist()) - set(measured))
        if unmeasured:
            raise ValueError(
                f"PUB {index}: observables act on qubits {unmeasured}, which the circuit does not measure."
            )

        param_shape = pub.parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, pub.observables)
        param_obs_map = defaultdict(set)
        for idx in np.ndindex(*bc_param_ind.shape):
            param_obs_map[bc_param_ind[idx]].update(bc_obs[idx])

        groupings: Dict[Tuple[str, ...], tuple] = {}
        metadata, obs_codes = [], []
        for param_index in np.ndindex(param_shape):  # binding (ravel) order == loop and push order
            labels = tuple(sorted(param_obs_map[param_index]))
            if labels not in groupings:
                groupings[labels] = _group_observables(labels, measured, options.abelian_grouping)
            codes, paulis = groupings[labels]
            obs_codes.append(codes)
            metadata.extend(
                {"meas_paulis": meas, "orig_paulis": orig, "param_index": param_index} for meas, orig in paulis
            )
        distinct_codes = {tuple(codes) for codes, _ in groupings.values()}
        # Same basis codes for every binding: one row, sent (or preloaded) once.
        rows = pack_obs_rows(list(distinct_codes) if len(distinct_codes) == 1 else obs_codes, len(measured))

        circuit_table, param_table = build_pub_tables(circuit, options.input_type, f"param_table_{tag}")
        obs_names = {f"obs_{j}" for j in range(len(measured))}
        obs_table = ParameterTable.from_qiskit(
            circuit,
            filter_function=lambda v: circuit.has_var(v) and v.name in obs_names,
            name=f"obs_table_{tag}",
        )
        obs_row = None
        if options.input_type is not None:
            obs_row = Parameter(
                f"obs_row_{tag}",
                value=[0] * len(rows[0]),
                qua_type=int,
                input_type=options.input_type,
                direction=Direction.INCOMING,
            )
        return cls(
            pub,
            index,
            circuit_table,
            param_table,
            metadata=metadata,
            param_indices=param_indices,
            active_qubits=[circuit.qubits[q] for q in measured],
            obs_codes=obs_codes,
            obs_table=obs_table,
            obs_row=obs_row,
            obs_rows=rows,
            words_per_obs=obs_words(len(measured)),
        )

    @property
    def shared_obs_row(self) -> bool:
        return len(self.obs_rows) == 1

    @property
    def streams_obs(self) -> bool:
        return self.obs_row is not None

    @property
    def streams_inputs(self) -> bool:
        return self.streams_params or self.streams_obs

    @property
    def shots(self) -> int:
        return int(np.ceil(1 / self.pub.precision**2))

    @property
    def num_qubits(self) -> int:
        return len(self.active_qubits)

    # ---------------------------------------------------------------------------------- QUA
    @property
    def obs_vars(self) -> list:
        return [self.obs_table.table[f"obs_{j}"].var for j in range(self.num_qubits)]

    def declare_observables(self) -> None:
        """QUA macro: declare the observable inputs, their delivery and scratch variables.

        A streamed row shared by every binding is received here, once, before the binding loop.
        """
        self.obs_table.declare()
        if self.streams_obs:
            self.obs_row.declare()
        else:
            self._obs_rows_qua = QUA2DArray(f"obs_rows_{self.obs_table.name}", np.array(self.obs_rows), qua_type=int)
            self._obs_rows_qua.declare()
        self._obs_word, self._obs_idx, self._obs_counter = declare(int), declare(int), declare(int)
        if self.streams_obs and self.shared_obs_row:
            rcv_obs_row(self.obs_row, self.words_per_obs, self._obs_counter)

    def _binding_obs_row(self, p):
        """QUA macro: the observable row of binding ``p``, received first when streamed per binding."""
        if self.streams_obs:
            if not self.shared_obs_row:
                rcv_obs_row(self.obs_row, self.words_per_obs, self._obs_counter)
            return self.obs_row.var
        return self._obs_rows_qua[0 if self.shared_obs_row else p]

    @contextmanager
    def observable_groups(self, p):
        """QUA macro: loop over binding ``p``'s measurement groups, each unpacked into ``obs_j``."""
        row = self._binding_obs_row(p)
        with for_(self._obs_idx, 0, self._obs_idx < row[0], self._obs_idx + 1):
            assign_obs_codes(row, 1 + self._obs_idx * self.words_per_obs, self.obs_vars, self._obs_word)
            yield

    # --------------------------------------------------------------------------------- host
    def _pushes(self) -> Iterator[Tuple[Any, Any]]:
        if self.streams_obs and self.shared_obs_row:
            yield self.obs_row, self.obs_rows[0]
        if self.streams_params:
            yield from super()._pushes()
        elif self.streams_obs and not self.shared_obs_row:
            for p in range(self.num_bindings):
                yield from self._binding_extras(p)

    def _binding_extras(self, p: int) -> Iterator[Tuple[Any, Any]]:
        if self.streams_obs and not self.shared_obs_row:
            yield self.obs_row, self.obs_rows[p]

    def _push_one(self, target, value, qm_job, qm) -> None:
        if target is self.obs_row:
            push_obs_row(target, value, self.words_per_obs, qm_job, qm)
        else:
            super()._push_one(target, value, qm_job, qm)

    def _hook_value(self, target: Parameter, value):
        if target is self.obs_row and target.input_type in (InputType.IO1, InputType.IO2):
            return value[: 1 + value[0] * self.words_per_obs]  # IO exchanges only the used prefix
        return value
