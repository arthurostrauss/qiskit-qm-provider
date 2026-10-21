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

"""How primitive PUB parameter bindings reach the QUA program.

Each parameterized PUB has two tables:

* the **circuit table** -- the fields the compiled circuit reads (``ParameterTable.from_qiskit``
  layout: full-length arrays for ``ParameterVector`` s in array mode, scalars otherwise);
* the **delivery table** -- what the client pushes.

For ``INPUT_STREAM`` / ``IO1`` / ``IO2`` the delivery table is a single ``<name>_row`` array
of ``len(circuit.parameters)`` floats: each binding is pushed as the raw
``parameter_values.ravel().as_array(circuit.parameters)`` row (one push, one advance) and
scattered into the circuit table in QUA. Unused vector elements are never streamed; they
keep the zero they are declared with. Compile-time sweeps (``input_type=None``) and OPNIC
(already one struct packet per table) deliver through the circuit table itself.

OPNIC is excluded only until it can be exercised: the intended contract is the same dense
row, i.e. one struct with a single array field, not one field per parameter. Enabling it means
adding OPNIC to ``_DENSE_INPUT_TYPES`` (mind Quarc's promotion of one-field structs to
standalone parameters).
"""

from __future__ import annotations

from typing import Optional, Tuple

from qm.qua import assign, fixed

from ..parameter_table import Direction, InputType, Parameter, ParameterTable
from ..parameter_table.qiskit_classical import (
    packed_qiskit_parameter_rows,
    qiskit_parameter_packer,
    qiskit_row_layout,
)

_DENSE_INPUT_TYPES = (InputType.INPUT_STREAM, InputType.IO1, InputType.IO2)


def build_pub_tables(
    circuit, input_type: Optional[InputType], name: str
) -> Tuple[Optional[ParameterTable], Optional[ParameterTable]]:
    """Return ``(circuit_table, delivery_table)`` for a PUB's circuit parameters (``None`` if it has none)."""
    from qiskit.circuit import Parameter as QiskitParameter

    dense = input_type in _DENSE_INPUT_TYPES
    circuit_table = ParameterTable.from_qiskit(
        circuit,
        input_type=None if dense else input_type,
        filter_function=lambda p: isinstance(p, QiskitParameter),
        name=name,
    )
    if circuit_table is None or not dense:
        return circuit_table, circuit_table
    row = Parameter(
        f"{name}_row",
        value=[0.0] * len(circuit.parameters),
        qua_type=fixed,
        input_type=input_type,
        direction=Direction.INCOMING,
    )
    delivery_table = ParameterTable([row], name=f"{name}_row")
    delivery_table._is_pub_row = True
    return circuit_table, delivery_table


def is_row_delivery(delivery_table) -> bool:
    """``True`` when ``delivery_table`` is a dense row table built by :func:`build_pub_tables`.

    The mode is an explicit marker on the table, never inferred from how the circuit and
    delivery tables relate, so a copied or separately built table is not mistaken for a row.
    """
    return getattr(delivery_table, "_is_pub_row", False)


def _row_parameter(circuit, circuit_table, delivery_table) -> Parameter:
    row = delivery_table.parameters[0]
    if circuit_table is delivery_table or row.length != len(circuit.parameters):
        raise ValueError(
            f"Row table {delivery_table.name!r} does not match the circuit: expected a separate "
            f"circuit table and a row of length {len(circuit.parameters)}, got {row.length}."
        )
    return row


def assign_qiskit_row(circuit, table: ParameterTable, row) -> None:
    """QUA macro: assign the ``row`` entries bound by ``circuit.parameters`` into ``table``.

    ``row`` is indexable by column (a QUA array or a ``QUA2DArray`` row). Only bound entries
    are assigned; unused vector elements keep their declared zero.
    """
    for column, key, index in qiskit_row_layout(circuit, table):
        var = table.table[key].var
        assign(var if index is None else var[index], row[column])


def load_pub_row(circuit, circuit_table: ParameterTable, delivery_table: ParameterTable) -> None:
    """QUA macro: receive one binding and make it visible to the circuit table."""
    delivery_table.rcv()
    if is_row_delivery(delivery_table):
        assign_qiskit_row(circuit, circuit_table, _row_parameter(circuit, circuit_table, delivery_table).var)


def push_pub_rows(pub, circuit_table, delivery_table, job, qm=None, after_row=None) -> None:
    """Push every binding of ``pub`` through ``delivery_table``; ``after_row(p)`` runs after row ``p``."""
    rows = pub.parameter_values.ravel().as_array(pub.circuit.parameters)
    if is_row_delivery(delivery_table):
        row_name = _row_parameter(pub.circuit, circuit_table, delivery_table).name
        values = ({row_name: row.tolist()} for row in rows)
    else:
        pack = qiskit_parameter_packer(pub.circuit, delivery_table)
        values = (pack(row) for row in rows)
    for p, value in enumerate(values):
        delivery_table.push_to_opx(value, job, qm)
        if after_row is not None:
            after_row(p)


def delivery_rows(pub, circuit_table, delivery_table) -> list[list]:
    """Per-binding values in ``delivery_table`` field order, for the cloud sync hook."""
    if is_row_delivery(delivery_table):
        _row_parameter(pub.circuit, circuit_table, delivery_table)
        return [[row] for row in pub.parameter_values.ravel().as_array(pub.circuit.parameters).tolist()]
    return packed_qiskit_parameter_rows(pub, delivery_table)
