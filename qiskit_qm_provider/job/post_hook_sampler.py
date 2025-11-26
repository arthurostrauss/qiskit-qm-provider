from qiskit.primitives.containers.sampler_pub import SamplerPub
from typing import List
import numpy as np
from qiskit_qm_provider.parameter_table import InputType, ParameterTable
from qm.qua import fixed

def generate_sync_hook_sampler(pubs: List[SamplerPub], parameter_tables: List[ParameterTable]) -> str:
    """Generate the sync hook code for the sampler."""
    
    parameter_value_dicts = np.array([pub.parameter_values.ravel().as_array([param.name for param in pub.circuit.parameters]) for pub in pubs])
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
                    input_type_str = f'InputType.{param.input_type.name}'
                
                # Serialize direction
                if param.input_type is None or param.input_type != InputType.DGX_Q:
                    direction_str = "None"
                elif param.direction is None:
                    direction_str = "None"
                else:
                    direction_str = f'Direction.{param.direction.name}'
                
                # Serialize units
                units_str = repr(param.units) if param.units else '""'
                
                # Build Parameter initialization string
                param_init = f"QMParameter(name={repr(param.name)}, value={value_str}, qua_type={qua_type_str}, input_type={input_type_str}, direction={direction_str}, units={units_str})"
                param_init_strings.append(param_init)
            
            # Build ParameterTable initialization string
            table_name = repr(parameter_table.name)
            param_list_str = "[" + ", ".join(param_init_strings) + "]"
            table_init_str = f"ParameterTable(parameters_dict={param_list_str}, name={table_name})"
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
        param_dict = {{param.name: value for param, value in zip(parameter_table.parameters, parameter_value)}}
        parameter_table.push_to_opx(param_dict, job)

# results_handle = job.result_handles
# results_handle.wait_for_all_values()

# all_data = []
# for i, circuit in enumerate(circuits):
#     qc_meas_data = {{}}
#     for creg in circuit.cregs:
#         data = results_handle.get("" + creg.name + "_" + str(i)).fetch_all()["value"]
#         meas_level = self.metadata.get("meas_level")
#         if meas_level == "classified":
#             bit_array = BitArray.from_samples(data.tolist(), creg.size).reshape(pub.shape)
#             qc_meas_data[creg.name] = bit_array
#         elif meas_level == "kerneled":
#             # TODO: Assume that buffering was done like (2, creg.size)
#             qc_meas_data[creg.name] = np.array([d[0] + 1j * d[1] for d in data], dtype=complex).reshape(
#                 pub.shape + (pub.shots, creg.size)
#             )
#         else:
#             # TODO: Figure it out
#             qc_meas_data[creg.name] = np.array([d[0] + 1j * d[1] for d in data], dtype=complex).reshape(
#                 pub.shape + (pub.shots, creg.size)
#             )

#     sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
#     all_data.append(sampler_data)

# result = PrimitiveResult(all_data)
# print(result)"""

    return sync_hook_code