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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple

import numpy as np
from qiskit.quantum_info import PauliList
from qm.qua import declare

from ..backend.backend_utils import estimator_measured_qubits, get_non_trivial_observables
from ..parameter_table import Direction, InputType, Parameter, ParameterTable, QUA2DArray
from .pub_parameters import (
    assign_obs_codes,
    assign_qiskit_row,
    build_pub_tables,
    delivery_rows,
    load_pub_row,
    obs_words,
    pack_obs_rows,
    push_obs_row,
    rcv_obs_row,
)
from ._sync_hook_common import _qua_type_str, serialize_table

if TYPE_CHECKING:
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from qiskit.primitives.containers.sampler_pub import SamplerPub
    from ..primitives.qm_estimator import QMEstimatorOptions


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
        self._param_values_qua: Optional[QUA2DArray] = None

    @property
    def num_bindings(self) -> int:
        return self.pub.parameter_values.ravel().size

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

    def sync_hook_pushes(self) -> List[list]:
        """Plain ``[input_type, name, qua_type, value]`` pushes, in order, for the IQCC sync hook."""
        pushes = []
        for target, value in self._pushes():
            if isinstance(target, ParameterTable):
                spec = serialize_table(target)
                for field_spec, field_value in zip(spec["params"], value):
                    pushes.append([spec["input_type"], field_spec["name"], field_spec["qua_type"], field_value])
            else:
                pushes.append([target.input_type.value, target.name, _qua_type_str(target), self._hook_value(target, value)])
        return pushes

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


def _measured_qubit_indices(circuit, observables) -> List[int]:
    """Qubits read into ``__c`` in clbit order (``__c[j]`` <-> ``obs_j``), as set by validation."""
    creg = next((creg for creg in circuit.cregs if creg.name == "__c"), None)
    if creg is None:
        return [circuit.find_bit(q).index for q in estimator_measured_qubits(circuit, observables)]
    measured = [None] * len(creg)
    for instruction in circuit.data:
        if instruction.operation.name == "measure" and instruction.clbits[0] in creg:
            measured[creg.index(instruction.clbits[0])] = circuit.find_bit(instruction.qubits[0]).index
    if any(q is None for q in measured):
        raise ValueError("Estimator circuit's '__c' register is not fully measured.")
    return measured


@dataclass(eq=False)
class EstimatorPlan(PrimitivePlan):
    """Plan for one :class:`~qiskit.primitives.containers.estimator_pub.EstimatorPub`.

    Grouping is done once per binding, on the observables restricted to the measured qubits
    (``__c[j]`` measures ``active_qubits[j]``, rotated by ``obs_j``), so ``metadata``, the
    observable codes and the ``__c`` stream share one order: binding-major, groups in order.
    """

    metadata: List[dict] = field(default_factory=list)
    param_indices: Optional[np.ndarray] = None
    active_qubits: List[Any] = field(default_factory=list)
    obs_codes: List[List[Tuple[int, ...]]] = field(default_factory=list)
    obs_table: Optional[ParameterTable] = None
    obs_row: Optional[Parameter] = None
    obs_rows: List[List[int]] = field(default_factory=list)
    words_per_obs: int = 1
    param_group_keys: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self._obs_rows_qua: Optional[QUA2DArray] = None
        self._obs_scratch: Optional[Tuple[Any, Any, Any]] = None

    @classmethod
    def from_pub(
        cls, pub: "EstimatorPub", options: "QMEstimatorOptions", plan_index: Optional[int] = None
    ) -> "EstimatorPlan":
        """Plan ``pub``; ``plan_index`` (its position in the job) names its tables uniquely."""
        circuit = pub.circuit
        tag = plan_index if plan_index is not None else circuit.name
        param_shape = pub.parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, pub.observables)
        param_obs_map = defaultdict(set)
        for index in np.ndindex(*bc_param_ind.shape):
            param_obs_map[bc_param_ind[index]].update(bc_obs[index])

        measured = _measured_qubit_indices(circuit, pub.observables)
        metadata, obs_codes, keys = [], [], []
        for param_index in np.ndindex(param_shape):  # binding (ravel) order == loop and push order
            keys.append(param_index)
            labels = sorted(param_obs_map[param_index])
            restricted = get_non_trivial_observables(PauliList(labels), measured).to_labels() if labels else []
            originals = defaultdict(list)
            for label, restricted_label in zip(labels, restricted):
                originals[restricted_label].append(label)
            unique = PauliList(list(originals)) if originals else None
            if unique is None:
                groups = []
            elif options.abelian_grouping:
                groups = unique.group_commuting(qubit_wise=True)
            else:
                groups = [unique[[i]] for i in range(len(unique))]
            codes = []
            for group in groups:
                z, x = np.logical_or.reduce(group.z), np.logical_or.reduce(group.x)
                codes.append(tuple(_basis_code(z[j], x[j]) for j in range(len(measured))))
                group_labels = group.to_labels()
                metadata.append(
                    {
                        "meas_paulis": PauliList([r for r in group_labels for _ in originals[r]]),
                        "orig_paulis": PauliList([o for r in group_labels for o in originals[r]]),
                        "param_index": param_index,
                    }
                )
            obs_codes.append(codes)

        circuit_table, param_table = build_pub_tables(circuit, options.input_type, f"param_table_{tag}")
        obs_names = {f"obs_{j}" for j in range(len(measured))}
        has_obs_inputs = all(circuit.has_var(name) for name in obs_names)
        obs_table = obs_row = None
        rows = pack_obs_rows(obs_codes, len(measured))
        if all(row == rows[0] for row in rows):
            rows = rows[:1]  # same observables for every binding: one shared row
        if has_obs_inputs:
            if options.input_type == InputType.OPNIC:
                raise NotImplementedError(
                    "Estimator observables over OPNIC are not supported yet; use INPUT_STREAM, IO1/IO2 "
                    "or compile-time (input_type=None)."
                )
            obs_table = ParameterTable.from_qiskit(
                circuit,
                filter_function=lambda v: circuit.has_var(v) and v.name in obs_names,
                name=f"obs_table_{tag}",
            )
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
            plan_index if plan_index is not None else 0,
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
            param_group_keys=keys,
        )

    # ------------------------------------------------------------------------ compatibility
    @property
    def obs_indices(self) -> List[List[Tuple[int, ...]]]:
        """Per binding, the per-qubit basis codes of each measurement group (alias of ``obs_codes``)."""
        return self.obs_codes

    @property
    def observables_var(self) -> Optional[ParameterTable]:
        """Alias of ``obs_table`` (the circuit's ``obs_j`` inputs)."""
        return self.obs_table

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
    def total_tasks(self) -> int:
        return len(self.metadata)

    @property
    def stream_buffer_count(self) -> int:
        """Number of ``(shots, n_bits)`` buffers saved to the ``__c`` stream (one per metadata task)."""
        return len(self.metadata)

    @property
    def uses_compile_time_param_bindings(self) -> bool:
        return self.uses_compile_time_params

    def stream_indices_for_metadata(self) -> List[int]:
        """Metadata and ``__c`` buffers share one order, so task ``k`` reads buffer ``k``."""
        return list(range(len(self.metadata)))

    @property
    def shots(self) -> int:
        return int(np.ceil(1 / self.pub.precision**2))

    @property
    def num_qubits(self) -> int:
        return len(self.active_qubits)

    @property
    def active_qubit_indices(self) -> List[int]:
        return [self.pub.circuit.find_bit(q).index for q in self.active_qubits]

    # ---------------------------------------------------------------------------------- QUA
    @property
    def obs_vars(self) -> list:
        return [self.obs_table.table[f"obs_{j}"].var for j in range(self.num_qubits)]

    def declare_observables(self) -> None:
        """QUA macro: declare the observable inputs, their delivery and scratch variables."""
        if self.obs_table is None:
            return
        self.obs_table.declare()
        if self.obs_row is not None:
            self.obs_row.declare()
        else:
            self._obs_rows_qua = QUA2DArray(f"obs_rows_{self.obs_table.name}", np.array(self.obs_rows), qua_type=int)
            self._obs_rows_qua.declare()
        self._obs_scratch = (declare(int), declare(int), declare(int))  # word, observable index, IO counter

    def load_obs_row(self, p, before_loop: bool):
        """QUA macro: receive (streamed) or select (compile-time) the observable row of binding ``p``.

        Called once before the binding loop (``before_loop=True``) and once per binding; a
        shared row is received only before the loop. Returns the row (indexable by QUA ints)
        or ``None`` when nothing is to be done at this call site.
        """
        if self.obs_table is None:
            return None
        _, _, counter = self._obs_scratch
        if self.obs_row is not None:
            if before_loop == self.shared_obs_row:
                rcv_obs_row(self.obs_row, self.words_per_obs, counter)
            return None if before_loop else self.obs_row.var
        if before_loop:
            return None
        return self._obs_rows_qua[0 if self.shared_obs_row else p]

    def assign_observable(self, row, obs_idx) -> None:
        """QUA macro: unpack observable ``obs_idx`` of ``row`` into the circuit's ``obs_j`` inputs."""
        word, _, _ = self._obs_scratch
        assign_obs_codes(row, 1 + obs_idx * self.words_per_obs, self.obs_vars, word)

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
