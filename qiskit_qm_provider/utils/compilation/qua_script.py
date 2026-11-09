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

"""Compile a circuit and return or dump the QUA script that would run.

Author: Arthur Strauss
Date: 2026-10-29
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from qiskit.circuit import QuantumCircuit
from qm import generate_qua_script

if TYPE_CHECKING:
    from ...backend.qm_backend import QMBackend


def get_qua_script(
    backend: "QMBackend",
    circuit: QuantumCircuit,
    param_table: Optional[Mapping[Any, Any]] = None,
) -> str:
    """Compile a circuit on its own and return the QUA script that would run, to inspect its
    pulse sequence.

    The circuit is compiled outside any QUA program, so no :class:`~.ParameterTable` or
    :class:`~.Parameter` can be declared for it: each circuit input instead starts from a plain
    Python value. Inputs missing from ``param_table`` start from a typed zero (``0.0``, ``0`` or
    ``False``, or a list of those for an array input), so a parameterized circuit needs no
    bindings at all. To compile with real streamed parameters, call
    :meth:`~.QMBackend.quantum_circuit_to_qua` inside ``with program():`` instead.

    Args:
        backend: The QM backend (used for quantum_circuit_to_qua and generate_config).
        circuit: The transpiled QuantumCircuit to compile to QUA.
        param_table: Optional Python values for some circuit inputs, keyed like a
            ``quantum_circuit_to_qua`` dict (circuit objects or their names).

    Returns:
        The QUA script string (Python source of the program that would be executed).

    Raises:
        TypeError: if ``param_table`` is not a dict (a table cannot be declared outside a
            QUA program).
    """
    from ...parameter_table.qiskit_classical import parameter_bindings, with_placeholders

    if param_table is not None and not isinstance(param_table, Mapping):
        raise TypeError(
            "get_qua_script compiles outside a QUA program, where tables cannot be declared: pass a "
            "dict of Python values, or omit it to start every input from zero"
        )
    _, exported = backend.qasm3_exporter.export(circuit)
    values = with_placeholders(exported, parameter_bindings(param_table, circuit))
    compilation = backend.quantum_circuit_to_qua(circuit, param_table=values)
    return generate_qua_script(compilation.qua_program, backend.generate_config())


def dump_qua_script(
    backend: "QMBackend",
    circuit: QuantumCircuit,
    path: str | None = None,
    param_table: Optional[Mapping[Any, Any]] = None,
) -> str:
    """Compile a circuit on its own and write the QUA script that would run to a file.

    See :func:`get_qua_script`: inputs start from the Python values in ``param_table``, or
    from a typed zero.

    Args:
        backend: The QM backend (used for quantum_circuit_to_qua and generate_config).
        circuit: The transpiled QuantumCircuit to compile to QUA.
        path: Output file path for the QUA script. If None, uses "debug_qua.py".
        param_table: Optional Python values for some circuit inputs (see
            :func:`get_qua_script`).

    Returns:
        The path to the written file.
    """
    qua_script = get_qua_script(backend, circuit, param_table=param_table)
    if path is None:
        path = "debug_qua.py"
    with open(path, "w") as f:
        f.write(qua_script)
    return path
