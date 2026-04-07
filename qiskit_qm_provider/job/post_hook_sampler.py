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

"""Sync-hook generation for sampler: serialize parameter tables for redeclaration in sync hooks.

Author: Arthur Strauss
Date: 2026-02-08
"""

from qiskit.primitives.containers.sampler_pub import SamplerPub
from typing import List
import numpy as np
from qiskit_qm_provider.parameter_table import InputType, ParameterTable
from qm.qua import fixed


def generate_sync_hook_sampler(
    pubs: List[SamplerPub], parameter_tables: List[ParameterTable]
) -> str:
    """Generate the sync hook code for the sampler."""

    parameter_value_dicts = np.array(
        [pub.parameter_values.ravel().as_array() for pub in pubs]
    )
    new_param_table_contents = []
    for parameter_table in parameter_tables:
        if parameter_table is None:
            new_param_table_contents.append(None)
        else:
            # Generate Parameter initialization strings for each parameter in the table
            param_init_strings = []
            for param in parameter_table.parameters:
                # Serialize the value
                if param.value is None:
                    value_str = "None"
                elif isinstance(param.value, (list, np.ndarray)):
                    # Convert array/list to proper Python literal
                    if isinstance(param.value, np.ndarray):
                        value_str = f"np.array({np.array2string(param.value, separator=', ', suppress_small=True)})"
                    else:
                        value_str = repr(param.value)
                elif isinstance(param.value, (int, float, bool)):
                    value_str = repr(param.value)
                else:
                    value_str = repr(param.value)

                # Serialize qua_type
                if param.type == int:
                    qua_type_str = "int"
                elif param.type == fixed:
                    qua_type_str = "fixed"
                elif param.type == bool:
                    qua_type_str = "bool"
                else:
                    qua_type_str = "None"

                # Serialize input_type
                if param.input_type is None:
                    input_type_str = "None"
                else:
                    input_type_str = f"InputType.{param.input_type.name}"

                # Serialize direction
                if param.input_type is None or param.input_type != InputType.OPNIC:
                    direction_str = "None"
                elif param.direction is None:
                    direction_str = "None"
                else:
                    direction_str = f"Direction.{param.direction.name}"

                # Serialize units
                units_str = repr(param.units) if param.units else '""'

                # Build Parameter initialization string
                param_init = f"QMParameter(name={repr(param.name)}, value={value_str}, qua_type={qua_type_str}, input_type={input_type_str}, direction={direction_str}, units={units_str})"
                param_init_strings.append(param_init)

            # Build ParameterTable initialization string
            table_name = repr(parameter_table.name)
            param_list_str = "[" + ", ".join(param_init_strings) + "]"
            table_init_str = (
                f"ParameterTable(parameters_dict={param_list_str}, name={table_name})"
            )
            new_param_table_contents.append(table_init_str)

    # Format parameter_tables list for insertion into sync_hook_code
    param_tables_list_str = "["
    for i, table_str in enumerate(new_param_table_contents):
        if i > 0:
            param_tables_list_str += ", "
        if table_str is None:
            param_tables_list_str += "None"
        else:
            param_tables_list_str += table_str
    param_tables_list_str += "]"

    sync_hook_code = f"""from iqcc_cloud_client.runtime import get_qm_job
from qiskit_qm_provider.parameter_table import ParameterTable, InputType, Parameter as QMParameter, Direction
from qm.qua import fixed

job = get_qm_job()

parameter_values = {np.array2string(parameter_value_dicts, separator=', ')}
parameter_tables = {param_tables_list_str}

for parameter_value, parameter_table in zip(parameter_values, parameter_tables):
    if parameter_table is not None and parameter_table.input_type is not None:
        for values in parameter_value:
            param_dict = {{param.name: float(value) for param, value in zip(parameter_table.parameters, values)}}
            parameter_table.push_to_opx(param_dict, job)
"""

    return sync_hook_code
