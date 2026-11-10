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

"""Base per-PUB execution plan: circuit parameters and streamed delivery.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ...parameter_table import Parameter, ParameterTable, QUA2DArray
from ..pub_parameters import (
    assign_qiskit_row,
    binding_rows,
    check_fixed_range,
    delivery_rows,
    load_pub_row,
)
from ..sync_hook import _qua_type_str, serialize_table

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
        """Validate parameter values and initialize compile-time binding storage."""
        check_fixed_range(self.binding_rows, self.pub.circuit.parameters, self.index)
        self._param_values_qua: Optional[QUA2DArray] = None

    @cached_property
    def binding_rows(self) -> np.ndarray:
        """All bindings as a ``(num_bindings, len(circuit.parameters))`` float matrix, built once."""
        return binding_rows(self.pub)

    @property
    def num_bindings(self) -> int:
        """Number of parameter bindings in this PUB (``pub.parameter_values.size``)."""
        return self.pub.parameter_values.size

    @property
    def streams_params(self) -> bool:
        """``True`` when circuit parameters are loaded via ``INPUT_STREAM``/``IO1``/``IO2``."""
        return self.param_table is not None and self.param_table.input_type is not None

    @property
    def uses_compile_time_params(self) -> bool:
        """``True`` when parameters are preloaded at compile time (``input_type=None``)."""
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
            self._param_values_qua = QUA2DArray(f"param_values_{self.param_table.name}", self.binding_rows)
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
        for p, values in enumerate(delivery_rows(self.pub, self.circuit_table, self.param_table, self.binding_rows)):
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
        """Push one streamed table row or scalar :class:`Parameter` to the running job."""
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
                yield specs[id(target)][0], value
