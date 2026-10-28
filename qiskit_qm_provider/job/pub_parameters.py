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

Every circuit parameter lands in a QUA ``fixed`` (signed 4.28 fixed point), so each bound value
must lie in ``[FIXED_MIN, FIXED_MAX]``; :func:`check_fixed_range` rejects a PUB otherwise.

Estimator observables travel the same way: per binding, one int row
``[n_obs, words(obs_0), ..., words(obs_{L-1})]`` padded to the plan's largest ``L``, where each
observable's per-qubit basis codes (0: Z/I, 1: X, 2: Y) are packed 2 bits per qubit,
``OBS_QUBITS_PER_WORD`` qubits per 32-bit word (the sign bit is never used). QUA unpacks a
row into the circuit's scalar ``obs_q`` inputs with shifts and masks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from qm.qua import IO1, IO2, assign, fixed, for_, pause

from ..parameter_table import Direction, InputType, Parameter, ParameterTable
from ..parameter_table.qiskit_classical import (
    packed_qiskit_parameter_rows,
    qiskit_parameter_packer,
    qiskit_row_layout,
)

_DENSE_INPUT_TYPES = (InputType.INPUT_STREAM, InputType.IO1, InputType.IO2)

FIXED_MIN = -8.0
FIXED_MAX = 8.0 - 2.0**-28
"""Range of QUA ``fixed`` (signed 4.28 fixed point): every circuit parameter value must lie in it."""


def binding_rows(pub) -> np.ndarray:
    """The PUB's bindings as a ``(num_bindings, len(circuit.parameters))`` float matrix (ravel order)."""
    return np.asarray(pub.parameter_values.ravel().as_array(pub.circuit.parameters), dtype=float)


def check_fixed_range(values: np.ndarray, parameters, index) -> None:
    """Reject bindings ``values`` (see :func:`binding_rows`) outside ``[FIXED_MIN, FIXED_MAX]`` or not finite.

    Circuit parameters are delivered and stored as QUA ``fixed`` on every path (compile-time
    rows, dense streamed rows, packed tables), and a value outside that range cannot be
    represented. Such values are not supported: rescale the parameter instead (see the
    *Parameter values and fixed-point range* section of the primitives guide).
    """
    bad = ~((values >= FIXED_MIN) & (values <= FIXED_MAX))
    if bad.any():
        binding, column = (int(i) for i in np.argwhere(bad)[0])
        raise ValueError(
            f"PUB {index}: parameter {parameters[column].name!r} has value {float(values[binding, column])!r} "
            f"(binding {binding}; {int(bad.sum())} out-of-range value(s) in total), outside the QUA "
            f"fixed-point range [{FIXED_MIN}, 8) used for circuit parameters. Values beyond this range "
            "are not supported: rescale the parameter so that every bound value fits."
        )


class PubRowTable(ParameterTable):
    """Delivery table of a PUB whose bindings travel as dense rows.

    It holds a single ``fixed`` array, :attr:`row`, of ``len(circuit.parameters)`` values: each
    binding is pushed as one raw ``as_array(circuit.parameters)`` row and scattered into the
    circuit table in QUA. Being a dedicated type (rather than a flag on a plain table) keeps the
    delivery mode explicit and survives copies. The constructor is the base one, so that
    :meth:`ParameterTable.__deepcopy__` can re-initialise a copy; build one with
    :meth:`for_circuit`.
    """

    @classmethod
    def for_circuit(cls, circuit, input_type: InputType, name: str) -> "PubRowTable":
        """The row table ``<name>_row`` for ``circuit``'s parameters, delivered through ``input_type``."""
        row = Parameter(
            f"{name}_row",
            value=[0.0] * len(circuit.parameters),
            qua_type=fixed,
            input_type=input_type,
            direction=Direction.INCOMING,
        )
        return cls([row], name=f"{name}_row")

    @property
    def row(self) -> Parameter:
        """The single array field carrying one binding."""
        return self.parameters[0]

    def check_circuit(self, circuit, circuit_table) -> Parameter:
        """Return :attr:`row` after checking it belongs to ``circuit`` and a separate circuit table."""
        if circuit_table is self or self.row.length != len(circuit.parameters):
            raise ValueError(
                f"Row table {self.name!r} does not match the circuit: expected a separate circuit "
                f"table and a row of length {len(circuit.parameters)}, got {self.row.length}."
            )
        return self.row


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
    return circuit_table, PubRowTable.for_circuit(circuit, input_type, name)


def is_row_delivery(delivery_table) -> bool:
    """``True`` when ``delivery_table`` is a :class:`PubRowTable` (bindings arrive as dense rows).

    Decided by type, never by name or by how the circuit and delivery tables relate, so a
    plain table that happens to look like a row is not mistaken for one.
    """
    return isinstance(delivery_table, PubRowTable)


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
        assign_qiskit_row(circuit, circuit_table, delivery_table.check_circuit(circuit, circuit_table).var)


def delivery_rows(pub, circuit_table, delivery_table, rows: Optional[np.ndarray] = None) -> list[list]:
    """Per-binding values in ``delivery_table`` field order (``rows``: :func:`binding_rows`, if at hand)."""
    if is_row_delivery(delivery_table):
        delivery_table.check_circuit(pub.circuit, circuit_table)
        return [[row] for row in (binding_rows(pub) if rows is None else rows).tolist()]
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


def obs_row_used_length(n_obs, words_per_obs: int):
    """Used prefix of an observable row, ``1 + n_obs * words_per_obs``: the count plus the words of
    its ``n_obs`` observables. ``n_obs`` is a Python int on the host and a QUA int in the program,
    so both sides of an IO exchange bound it with this one expression."""
    return 1 + n_obs * words_per_obs


def _is_io(param: Parameter) -> bool:
    return param.input_type in (InputType.IO1, InputType.IO2)


def obs_row_payload(row_param: Parameter, row: list[int], words_per_obs: int) -> list[int]:
    """What the host sends for ``row``: the whole row on an input stream, the used prefix over IO."""
    row = [int(v) for v in row]
    return row[: obs_row_used_length(row[0], words_per_obs)] if _is_io(row_param) else row


def rcv_obs_row(row_param: Parameter, words_per_obs: int, counter) -> None:
    """QUA macro: receive one observable row (the counterpart of :func:`obs_row_payload`).

    Input streams advance once (the row is pushed whole). Over IO only the used prefix is
    exchanged, one pause per word, so padding costs nothing.
    """
    if _is_io(row_param):
        io = IO1 if row_param.input_type == InputType.IO1 else IO2
        pause()
        assign(row_param.var[0], io)
        with for_(counter, 1, counter < obs_row_used_length(row_param.var[0], words_per_obs), counter + 1):
            pause()
            assign(row_param.var[counter], io)
    else:
        row_param.rcv()


def push_obs_row(row_param: Parameter, payload: list[int], job, qm=None) -> None:
    """Send an :func:`obs_row_payload` to a running job (IO payloads may be shorter than the row)."""
    if _is_io(row_param):
        row_param.push_io_values(payload, job, qm)
    else:
        row_param.push_to_opx(payload, job, qm)
