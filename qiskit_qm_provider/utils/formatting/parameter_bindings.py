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

"""Turn a ``param_table`` / ``table_by_ref`` argument into field-name bindings.

Author: Arthur Strauss
Date: 2026-11-09
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from qiskit.circuit import Parameter as QiskitParameter, ParameterVector, QuantumCircuit
from qiskit.circuit.classical.expr import Var
from qiskit.circuit.parametervector import ParameterVectorElement

from ...parameter_table import Parameter, ParameterTable
from ...parameter_table.qiskit_classical import qiskit_input_key, vector_element_key


def parameter_bindings(param_table, qc: QuantumCircuit) -> Dict[str, Any]:
    """Field name to QUA value from a ``param_table`` / ``table_by_ref`` argument."""
    if param_table is None:
        return {}
    if isinstance(param_table, (ParameterTable, Parameter)):
        param_table = [param_table]
    bindings: Dict[str, Any] = {}
    if isinstance(param_table, Sequence):
        for table in param_table:
            if not table.is_declared:
                table.declare(pause_program=False)
            variables = table.variables_dict if isinstance(table, ParameterTable) else {table.name: table.var}
            bindings.update(variables)
        return bindings
    if isinstance(param_table, Dict):
        named_inputs = {p.vector.name: p.vector for p in qc.parameters if isinstance(p, ParameterVectorElement)}
        named_inputs.update((p.name, p) for p in qc.parameters)
        for key, value in param_table.items():
            if isinstance(key, str):
                key = named_inputs.get(key, key)
            if isinstance(key, ParameterVectorElement):
                bindings[vector_element_key(key)] = value
            elif isinstance(key, (ParameterVector, QiskitParameter, Var)):
                bindings[qiskit_input_key(key)] = value
            else:
                bindings[key] = value
        return bindings
    raise TypeError(
        "parameter mapping must be a Parameter, ParameterTable, a sequence of either, or a dict of bindings"
    )
