from typing import List, Optional
import numpy as np
from qiskit_qm_provider.parameter_table import InputType, ParameterTable
from qm.qua import fixed
from .qm_estimator_job import _ExecutionPlan


def _serialize_parameter_table(parameter_table: Optional[ParameterTable]) -> str:
    """Serialize a ParameterTable to a string representation for redeclaration."""
    if parameter_table is None:
        return "None"
    
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
    return table_init_str


def _serialize_obs_indices(obs_indices: List[List[tuple]]) -> str:
    """Serialize obs_indices (List[List[Tuple[int, ...]]]) to a Python list string."""
    # Convert to nested list representation
    result = "["
    for i, inner_list in enumerate(obs_indices):
        if i > 0:
            result += ", "
        result += "["
        for j, obs_tuple in enumerate(inner_list):
            if j > 0:
                result += ", "
            result += repr(obs_tuple)
        result += "]"
    result += "]"
    return result


def generate_sync_hook_estimator(execution_plans: List[_ExecutionPlan]) -> str:
    """Generate the sync hook code for the estimator."""
    
    # Extract parameter values for each execution plan
    parameter_values_list = []
    obs_indices_list = []
    param_table_contents = []
    observables_var_contents = []
    
    for plan in execution_plans:
        # Serialize parameter values
        if plan.param_table is not None and len(plan.param_table.parameters) > 0:
            # Extract parameter values as array, matching QMEstimatorJob.submit() pattern
            param_values = plan.pub.parameter_values.ravel().as_array()
            parameter_values_list.append(param_values)
        else:
            # No parameters, use empty list
            parameter_values_list.append([])
        
        # Serialize obs_indices
        obs_indices_list.append(plan.obs_indices)
        
        # Serialize parameter tables
        param_table_str = _serialize_parameter_table(plan.param_table)
        param_table_contents.append(param_table_str)
        
        observables_var_str = _serialize_parameter_table(plan.observables_var)
        observables_var_contents.append(observables_var_str)
    
    # Format parameter_values list for insertion into sync_hook_code
    param_values_list_str = "["
    for i, param_values in enumerate(parameter_values_list):
        if i > 0:
            param_values_list_str += ", "
        if len(param_values) == 0:
            param_values_list_str += "[]"
        else:
            # Convert to list of arrays for proper serialization
            param_values_str = "["
            for j, pv in enumerate(param_values):
                if j > 0:
                    param_values_str += ", "
                if isinstance(pv, np.ndarray):
                    param_values_str += f"np.array({np.array2string(pv, separator=', ')})"
                else:
                    param_values_str += repr(pv)
            param_values_str += "]"
            param_values_list_str += param_values_str
    param_values_list_str += "]"
    
    # Format obs_indices list
    obs_indices_list_str = "["
    for i, obs_indices in enumerate(obs_indices_list):
        if i > 0:
            obs_indices_list_str += ", "
        obs_indices_list_str += _serialize_obs_indices(obs_indices)
    obs_indices_list_str += "]"
    
    # Format param_tables list
    param_tables_list_str = "["
    for i, table_str in enumerate(param_table_contents):
        if i > 0:
            param_tables_list_str += ", "
        param_tables_list_str += table_str
    param_tables_list_str += "]"
    
    # Format observables_vars list
    observables_vars_list_str = "["
    for i, table_str in enumerate(observables_var_contents):
        if i > 0:
            observables_vars_list_str += ", "
        observables_vars_list_str += table_str
    observables_vars_list_str += "]"
    
    sync_hook_code = f"""from iqcc_cloud_client.runtime import get_qm_job
from qiskit_qm_provider.parameter_table import ParameterTable, InputType, Parameter as QMParameter, Direction
from qm.qua import fixed
import numpy as np

job = get_qm_job()

parameter_values_list = {param_values_list_str}
obs_indices_list = {obs_indices_list_str}
param_tables = {param_tables_list_str}
observables_vars = {observables_vars_list_str}

for i in range(len(parameter_values_list)):
    param_table = param_tables[i]
    observables_var = observables_vars[i]
    parameter_values = parameter_values_list[i]
    obs_indices = obs_indices_list[i]
    
    # Push parameter values if the circuit has them
    if param_table is not None and param_table.input_type is not None:
        for p, param_value in enumerate(parameter_values):
            param_dict = {{param.name: value for param, value in zip(param_table.parameters, param_value)}}
            param_table.push_to_opx(param_dict, job)
            if observables_var.input_type is not None:
                for obs_value in obs_indices[p]:
                    obs_dict = {{f"obs_{{q}}": val for q, val in enumerate(obs_value)}}
                    observables_var.push_to_opx(obs_dict, job)
    # Push observable indices
    elif observables_var.input_type is not None:
        for obs_value in obs_indices[0]:
            obs_dict = {{f"obs_{{q}}": val for q, val in enumerate(obs_value)}}
            observables_var.push_to_opx(obs_dict, job)

# results_handle = job.result_handles
# results_handle.wait_for_all_values()

# Post-processing of results (commented out - handled on client side)
# Note: The following code structure shows how results would be processed,
# but requires execution_plans and methods from QMEstimatorJob which are not
# available in the sync hook context. Actual post-processing is done in
# IQCCEstimatorJob._result_function() on the client side.
# 
# pub_results = []
# for i in range(len(parameter_values_list)):
#     # Get bitstring data from result handles
#     # data = np.array(results_handle.get(f"__c_{{i}}")).flatten().tolist()
#     
#     # Reshape data: (total_tasks * shots, num_bits) -> (total_tasks, shots, num_bits)
#     # bitstrings = np.array(data).reshape(total_tasks, shots, num_qubits)
#     
#     # Compute expectation values from bitstrings
#     # (Requires _calc_expval_map method from QMEstimatorJob)
#     
#     # Postprocess to get PubResult with expectation values and standard errors
#     # (Requires _postprocess_pub method from QMEstimatorJob)
#     # pub_results.append(pub_result)
# 
# # result = PrimitiveResult(pub_results, metadata={{"version": 2}})
# # print(result)
"""

    return sync_hook_code

