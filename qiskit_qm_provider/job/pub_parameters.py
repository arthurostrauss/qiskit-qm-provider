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

"""How primitive PUB inputs (parameter bindings, estimator observables) reach the QUA program.

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

Estimator observables travel the same way: per binding, one int row
``[n_obs, words(obs_0), ..., words(obs_{L-1})]`` padded to the plan's largest ``L``, where each
observable's per-qubit basis codes (0: Z/I, 1: X, 2: Y) are packed 2 bits per qubit,
``OBS_QUBITS_PER_WORD`` qubits per 32-bit word (the sign bit is never used). QUA unpacks a
row into the circuit's scalar ``obs_q`` inputs with shifts and masks.
"""

from __future__ import annotations

from typing import Optional, Tuple

from qm.qua import IO1, IO2, assign, fixed, for_, pause
from qm.api.v2.job_api import JobApi
from qualang_tools.results import wait_until_job_is_paused

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


def delivery_rows(pub, circuit_table, delivery_table) -> list[list]:
    """Per-binding values in ``delivery_table`` field order, for the cloud sync hook."""
    if is_row_delivery(delivery_table):
        _row_parameter(pub.circuit, circuit_table, delivery_table)
        return [[row] for row in pub.parameter_values.ravel().as_array(pub.circuit.parameters).tolist()]
    return packed_qiskit_parameter_rows(pub, delivery_table)



# --------------------------------------------------------------------------------------------
# Estimator observable rows
# --------------------------------------------------------------------------------------------

OBS_QUBITS_PER_WORD = 15
"""Qubits packed per int word: 2 bits each in bits 0..29, keeping the sign bit clear."""


def obs_words(num_qubits: int) -> int:
    """Words needed for one observable's codes on ``num_qubits`` measured qubits (at least 1)."""
    return max(1, -(-num_qubits // OBS_QUBITS_PER_WORD))


def pack_obs_codes(codes) -> list[int]:
    """Pack per-qubit basis codes (0, 1, 2) into words: qubit ``q`` -> word ``q // 15``, bits ``2 * (q % 15)``."""
    words = [0] * obs_words(len(codes))
    for q, code in enumerate(codes):
        words[q // OBS_QUBITS_PER_WORD] |= int(code) << (2 * (q % OBS_QUBITS_PER_WORD))
    return words


def unpack_obs_codes(words, num_qubits: int) -> list[int]:
    """Inverse of :func:`pack_obs_codes` (host side)."""
    return [
        (int(words[q // OBS_QUBITS_PER_WORD]) >> (2 * (q % OBS_QUBITS_PER_WORD))) & 3 for q in range(num_qubits)
    ]


def pack_obs_rows(codes_per_binding, num_qubits: int) -> list[list[int]]:
    """One row ``[n_obs, words...]`` per binding, zero-padded to the longest binding."""
    width = obs_words(num_qubits)
    longest = max((len(codes) for codes in codes_per_binding), default=0)
    rows = []
    for codes in codes_per_binding:
        row = [len(codes)] + [w for obs in codes for w in pack_obs_codes(obs)]
        rows.append(row + [0] * (1 + longest * width - len(row)))
    return rows


def assign_obs_codes(row, base, obs_vars, word) -> None:
    """QUA macro: unpack one observable from ``row[base:]`` into the ``obs_vars`` (qubit order).

    ``row`` is indexable by QUA expressions (an input-stream array or a ``QUA2DArray`` row),
    ``base`` the observable's first word (int or QUA expression) and ``word`` a scratch QUA int.
    """
    for w in range(obs_words(len(obs_vars))):
        assign(word, row[base + w])
        for q in range(w * OBS_QUBITS_PER_WORD, min(len(obs_vars), (w + 1) * OBS_QUBITS_PER_WORD)):
            shift = 2 * (q % OBS_QUBITS_PER_WORD)
            assign(obs_vars[q], ((word >> shift) & 3) if shift else (word & 3))


def rcv_obs_row(row_param: Parameter, words_per_obs: int, counter) -> None:
    """QUA macro: receive one observable row.

    Input streams advance once (the row is pushed whole). Over IO only the used prefix
    ``1 + n_obs * words_per_obs`` is exchanged, one pause per word, so padding costs nothing.
    """
    if row_param.input_type in (InputType.IO1, InputType.IO2):
        io = IO1 if row_param.input_type == InputType.IO1 else IO2
        pause()
        assign(row_param.var[0], io)
        with for_(counter, 1, counter < 1 + row_param.var[0] * words_per_obs, counter + 1):
            pause()
            assign(row_param.var[counter], io)
    else:
        row_param.rcv()


def push_obs_row(row_param: Parameter, row: list[int], words_per_obs: int, job, qm=None, time_out: int = 90) -> None:
    """Host counterpart of :func:`rcv_obs_row`."""
    if row_param.input_type in (InputType.IO1, InputType.IO2):
        io = "io1" if row_param.input_type == InputType.IO1 else "io2"
        for value in row[: 1 + row[0] * words_per_obs]:
            wait_until_job_is_paused(job, time_out)
            (job if isinstance(job, JobApi) else qm).set_io_values(**{io: int(value)})
            job.resume()
    else:
        row_param.push_to_opx([int(v) for v in row], job, qm)
