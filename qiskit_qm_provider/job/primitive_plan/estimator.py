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

"""Estimator readout contract, grouping helpers, and per-PUB execution plan.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from qiskit.quantum_info import PauliList
from qm.qua import declare, for_

from ...parameter_table import Direction, InputType, Parameter, ParameterTable, QUA2DArray
from ..pub_parameters import (
    ObsRow,
    assign_obs_codes,
    build_pub_tables,
    obs_row_payload,
    obs_row_used_length,
    obs_words,
    push_obs_row,
    rcv_obs_row,
)
from .base import PrimitivePlan

if TYPE_CHECKING:
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from ...primitives.qm_estimator import QMEstimatorOptions

ESTIMATOR_RESERVED_PREFIX = "__"
"""Prefix of the classical names the estimator adds to a PUB circuit; user circuits must not use it."""

ESTIMATOR_READOUT_REGISTER = "__c"
"""Classical register whose bit ``j`` holds the outcome of the ``j``-th measured qubit."""


def estimator_basis_input(j: int) -> str:
    """Name of the ``Uint`` input selecting the measurement basis of the ``j``-th measured qubit."""
    return f"__obs_{j}"


def estimator_measured_qubits(observables) -> list[int]:
    """Indices of the qubits the estimator must measure for ``observables``.

    ``observables`` is anything :class:`~qiskit.primitives.containers.observables_array.ObservablesArray`
    can coerce, taken as ISA: it spans the circuit's full physical register (Qiskit's
    ``apply_layout`` convention). A qubit is measured iff some term of some observable in the
    array acts non-trivially on it, whether or not a gate touches it and whether or not it is an
    ancilla. Every other qubit only ever carries an identity, so it cannot change any
    expectation value and is not read. When every term is the identity, no qubit is measured:
    ``<I> = 1`` exactly, so those terms are evaluated without running the circuit.

    Raises:
        ValueError: If a term is not a Pauli string (e.g. a ``SparseObservable`` projector
            such as ``"0"``, ``"+"`` or ``"r"``), which the estimator does not support yet.
    """
    from qiskit.primitives.containers.observables_array import ObservablesArray

    observables = ObservablesArray.coerce(observables).ravel().tolist()
    labels = {label for observable in observables for label in observable if label.strip("I")}
    non_pauli = sorted(label for label in labels if label.strip("IXYZ"))
    if non_pauli:
        raise ValueError(
            f"Observable terms {non_pauli[:3]} are not Pauli strings. QMEstimatorV2 supports Pauli "
            "observables only; expand projector terms into Paulis first, e.g. with "
            "SparseObservable.as_paulis() (the number of terms doubles per projector)."
        )
    if not labels:
        return []
    paulis = PauliList(sorted(labels))
    return np.flatnonzero(np.logical_or.reduce(paulis.x | paulis.z, axis=0)).tolist()


def get_non_trivial_observables(observables: PauliList, active_qubit_indices: List[int]) -> PauliList:
    """Restrict observables to the given qubits.

    Args:
        observables: Pauli observables defined on the full qubit register.
        active_qubit_indices: Indices of the qubits to keep, in the order of the result.

    Returns:
        A :class:`~qiskit.quantum_info.PauliList` on ``len(active_qubit_indices)`` qubits whose
        qubit ``j`` is qubit ``active_qubit_indices[j]`` of the input; other qubits are dropped.
    """
    observables = PauliList(observables)
    indices = list(active_qubit_indices)
    return PauliList.from_symplectic(observables.z[:, indices], observables.x[:, indices], observables.phase)


def _basis_code(z: bool, x: bool) -> int:
    """Per-qubit measurement basis code consumed by the estimator's switch: 0 Z/I, 1 X, 2 Y."""
    return 0 if not x else (2 if z else 1)


def _estimator_readout(circuit) -> List[int]:
    """Check the estimator readout contract and return the measured qubits in ``__c`` order.

    :meth:`~qiskit_qm_provider.primitives.QMEstimatorV2.validate_estimator_pubs` prepares every
    PUB circuit this way: a ``__c`` register whose bit ``j`` measures qubit ``measured[j]``,
    preceded by a basis switch on that qubit driven by the input variable ``__obs_j``. The
    plan writes the per-group basis codes into ``__obs_j`` and reads the outcomes from ``__c``.
    """
    hint = "Prepare estimator PUBs with QMEstimatorV2.validate_estimator_pubs."
    creg = next((creg for creg in circuit.cregs if creg.name == ESTIMATOR_READOUT_REGISTER), None)
    if creg is None:
        raise ValueError(
            f"Estimator circuit {circuit.name!r} has no '{ESTIMATOR_READOUT_REGISTER}' readout register. {hint}"
        )
    measured = [None] * len(creg)
    for instruction in circuit.data:
        if instruction.operation.name == "measure" and instruction.clbits[0] in creg:
            measured[creg.index(instruction.clbits[0])] = circuit.find_bit(instruction.qubits[0]).index
    if any(q is None for q in measured):
        raise ValueError(f"Estimator circuit {circuit.name!r}: '__c' is not fully measured. {hint}")
    missing = [estimator_basis_input(j) for j in range(len(measured)) if not circuit.has_var(estimator_basis_input(j))]
    if missing:
        raise ValueError(f"Estimator circuit {circuit.name!r} lacks the basis inputs {missing}. {hint}")
    return measured


def _is_identity(label: str) -> bool:
    """Return whether ``label`` is an all-identity Pauli string (no measurement needed)."""
    return not label.strip("I")


def _group_observables(labels: Tuple[str, ...], measured: List[int], abelian_grouping: bool):
    """Measurement groups of one binding's non-identity Pauli terms.

    Returns the per-group basis codes (in ``__c`` order) and, per group, the
    ``(meas_paulis, orig_paulis)`` pair used to post-process its counts. ``labels`` must act
    trivially outside ``measured`` (checked here), so restriction to ``measured`` is one-to-one.
    """
    if not labels:
        return [], []
    paulis = PauliList(list(labels))
    outside = np.ones(paulis.num_qubits, dtype=bool)
    outside[measured] = False
    unmeasured = np.flatnonzero(np.logical_or.reduce((paulis.x | paulis.z)[:, outside], axis=0))
    if unmeasured.size:
        qubits = np.flatnonzero(outside)[unmeasured].tolist()
        raise ValueError(f"Observables act on qubits {qubits}, which the circuit does not measure.")
    restricted = get_non_trivial_observables(paulis, measured)
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
    Identity terms are never measured: ``<I> = 1`` exactly, so they only enter post-processing.
    The other terms are grouped per distinct set (shared by all bindings that request the same
    terms), restricted to the measured qubits (``__c[j]`` measures ``active_qubits[j]``,
    rotated by ``__obs_j``). ``metadata``, the observable codes and the ``__c`` stream share one
    order: binding-major, groups in order. A plan without any task (only identity terms) is
    neither compiled nor sent: see :attr:`has_tasks`.
    """

    metadata: List[dict] = field(default_factory=list)
    param_indices: Optional[np.ndarray] = None
    active_qubits: List[Any] = field(default_factory=list)
    obs_codes: List[List[Tuple[int, ...]]] = field(default_factory=list)
    obs_table: Optional[ParameterTable] = None
    obs_row: Optional[Parameter] = None
    obs_rows: List[ObsRow] = field(default_factory=list)
    words_per_obs: int = 1

    def __post_init__(self):
        """Initialize observable delivery storage and QUA scratch variables."""
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
        measured = _estimator_readout(circuit)

        param_shape = pub.parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, pub.observables)
        param_obs_map = defaultdict(set)
        for idx in np.ndindex(*bc_param_ind.shape):
            param_obs_map[bc_param_ind[idx]].update(bc_obs[idx])

        groupings: Dict[Tuple[str, ...], tuple] = {}
        metadata, obs_codes = [], []
        for param_index in np.ndindex(param_shape):  # binding (ravel) order == loop and push order
            labels = tuple(sorted(label for label in param_obs_map[param_index] if not _is_identity(label)))
            if labels not in groupings:
                try:
                    groupings[labels] = _group_observables(labels, measured, options.abelian_grouping)
                except ValueError as error:
                    raise ValueError(f"PUB {index}: {error}") from None
            codes, paulis = groupings[labels]
            obs_codes.append(codes)
            metadata.extend(
                {"meas_paulis": meas, "orig_paulis": orig, "param_index": param_index} for meas, orig in paulis
            )
        if not metadata:  # zero bindings or only identity terms: nothing to run on the device
            return cls(pub, index, None, None, param_indices=param_indices, obs_codes=obs_codes)
        distinct_codes = {tuple(codes) for codes, _ in groupings.values()}
        # Same basis codes for every binding: one row, sent (or preloaded) once.
        rows = ObsRow.pack(list(distinct_codes) if len(distinct_codes) == 1 else obs_codes, len(measured))

        circuit_table, param_table = build_pub_tables(circuit, options.input_type, f"param_table_{tag}")
        obs_names = {estimator_basis_input(j) for j in range(len(measured))}
        obs_table = ParameterTable.from_qiskit(
            circuit,
            filter_function=lambda v: circuit.has_var(v) and v.name in obs_names,
            name=f"obs_table_{tag}",
        )
        obs_row = None
        if options.input_type is not None:
            obs_row = Parameter(
                f"obs_row_{tag}",
                value=[0] * len(rows[0].wire),
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
    def has_tasks(self) -> bool:
        """``False`` when every term is the identity or the PUB has zero bindings: the PUB is then
        not compiled, sent or run (its values are exact, or its result is empty)."""
        return bool(self.metadata)

    @property
    def shared_obs_row(self) -> bool:
        """``True`` when every binding uses the same packed observable row (one send or preload)."""
        return len(self.obs_rows) == 1

    @property
    def streams_obs(self) -> bool:
        """``True`` when packed observable rows are loaded via ``INPUT_STREAM``/``IO1``/``IO2``."""
        return self.obs_row is not None

    @property
    def streams_inputs(self) -> bool:
        """``True`` when the client must push circuit parameters and/or observable rows."""
        return self.streams_params or self.streams_obs

    @property
    def shots(self) -> int:
        """Shot count implied by this PUB's target precision (``ceil(1 / precision²)``)."""
        return int(np.ceil(1 / self.pub.precision**2))

    @property
    def num_qubits(self) -> int:
        """Number of qubits read out through the ``__c`` register."""
        return len(self.active_qubits)

    # ---------------------------------------------------------------------------------- QUA
    @property
    def obs_vars(self) -> list:
        """QUA variables ``__obs_j`` that select the measurement basis per readout qubit."""
        return [self.obs_table.table[estimator_basis_input(j)].var for j in range(self.num_qubits)]

    def declare_observables(self) -> None:
        """QUA macro: declare the observable inputs, their delivery and scratch variables.

        A streamed row shared by every binding is received here, once, before the binding loop.
        """
        self.obs_table.declare()
        if self.streams_obs:
            self.obs_row.declare()
        else:
            self._obs_rows_qua = QUA2DArray(
                f"obs_rows_{self.obs_table.name}", np.array([row.wire for row in self.obs_rows]), qua_type=int
            )
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
        """QUA macro: loop over binding ``p``'s measurement groups, each unpacked into ``__obs_j``."""
        row = self._binding_obs_row(p)
        with for_(self._obs_idx, 0, self._obs_idx < row[0], self._obs_idx + 1):
            # Group g's words start right after the count and the first g groups.
            assign_obs_codes(row, obs_row_used_length(self._obs_idx, self.words_per_obs), self.obs_vars, self._obs_word)
            yield

    # --------------------------------------------------------------------------------- host
    def _pushes(self) -> Iterator[Tuple[Any, Any]]:
        """Push order: shared observable row (if any), then parameter rows and per-binding obs rows."""
        if self.streams_obs and self.shared_obs_row:
            yield self.obs_row, self._obs_payload(0)
        if self.streams_params:
            yield from super()._pushes()
        elif self.streams_obs and not self.shared_obs_row:
            for p in range(self.num_bindings):
                yield from self._binding_extras(p)

    def _binding_extras(self, p: int) -> Iterator[Tuple[Any, Any]]:
        """Per-binding packed observable row when rows differ across bindings."""
        if self.streams_obs and not self.shared_obs_row:
            yield self.obs_row, self._obs_payload(p)

    def _obs_payload(self, p: int) -> List[int]:
        """Row ``p`` as sent, for both the local push and the sync hook (see :meth:`ObsRow.payload`)."""
        return obs_row_payload(self.obs_row, self.obs_rows[p], self.words_per_obs)

    def _push_one(self, target, value, qm_job, qm) -> None:
        """Push an observable row or delegate to :meth:`~.base.PrimitivePlan._push_one`."""
        if target is self.obs_row:
            push_obs_row(target, value, qm_job, qm)
        else:
            super()._push_one(target, value, qm_job, qm)
