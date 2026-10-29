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

from typing import TYPE_CHECKING

from qiskit.circuit import QuantumCircuit
from qm import generate_qua_script

if TYPE_CHECKING:
    from ...backend.qm_backend import QMBackend


def get_qua_script(
    backend: "QMBackend",
    circuit: QuantumCircuit,
    param_table=None,
) -> str:
    """Compile a circuit to QUA and return the running QUA script as a string.

    Args:
        backend: The QM backend (used for quantum_circuit_to_qua and generate_config).
        circuit: The transpiled QuantumCircuit to compile to QUA.
        param_table: Optional parameter table for parameterized circuits
            (same as for quantum_circuit_to_qua).

    Returns:
        The QUA script string (Python source of the program that would be executed).
    """
    compilation_result = backend.quantum_circuit_to_qua(circuit, param_table=param_table)
    qua_program = compilation_result.result_program.dsl_program
    config = backend.generate_config()
    return generate_qua_script(qua_program, config)


def dump_qua_script(
    backend: "QMBackend",
    circuit: QuantumCircuit,
    path: str | None = None,
    param_table=None,
) -> str:
    """Compile a circuit to QUA and write the running QUA script to a file.

    Useful for inspecting the actual QUA program that would be executed
    when running the circuit on the backend.

    Args:
        backend: The QM backend (used for quantum_circuit_to_qua and generate_config).
        circuit: The transpiled QuantumCircuit to compile to QUA.
        path: Output file path for the QUA script. If None, uses "debug_qua.py".
        param_table: Optional parameter table for parameterized circuits
            (same as for quantum_circuit_to_qua).

    Returns:
        The path to the written file.
    """
    qua_script = get_qua_script(backend, circuit, param_table=param_table)
    if path is None:
        path = "debug_qua.py"
    with open(path, "w") as f:
        f.write(qua_script)
    return path
