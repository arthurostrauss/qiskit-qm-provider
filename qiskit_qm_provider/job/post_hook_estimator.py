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

"""Sync-hook generation for estimator jobs (e.g. IQCC cloud).

The emitted hook depends only on ``iqcc_cloud_client.runtime.get_qm_job`` — it does
not import ``qiskit_qm_provider`` or ``numpy``. Execution plans (parameters and
observable indices) are reduced to plain data here and the push logic is rendered
from a Jinja template.

Author: Arthur Strauss
Date: 2026-02-08
"""

from typing import List

from qiskit_qm_provider.parameter_table import Parameter as QuaParameter
from ._sync_hook_common import jinja_env, serialize_table, to_py_literal
from .qm_estimator_job import _ExecutionPlan


def _serialize_obs_indices(obs_indices: List[List[tuple]]) -> list:
    """Convert nested obs indices to plain Python lists of ``int``."""
    return [[[int(v) for v in obs_tuple] for obs_tuple in inner_list] for inner_list in obs_indices]


def generate_sync_hook_estimator(execution_plans: List[_ExecutionPlan], obs_length_var: QuaParameter) -> str:
    """Generate the sync hook code for the estimator."""
    plans_data = []
    for plan in execution_plans:
        param_table = serialize_table(plan.param_table)
        if param_table is not None:
            parameter_values = plan.pub.parameter_values.ravel().as_array().tolist()
        else:
            parameter_values = []
        plans_data.append(
            {
                "param_table": param_table,
                "observables_var": serialize_table(plan.observables_var),
                "parameter_values": parameter_values,
                "obs_indices": _serialize_obs_indices(plan.obs_indices),
            }
        )

    obs_length_var_data = {
        "name": obs_length_var.name,
        "input_type": obs_length_var.input_type.value if obs_length_var.input_type is not None else None,
        "qua_type": "int",
    }

    template = jinja_env().get_template("sync_hook_estimator.py.jinja")
    return template.render(
        plans_literal=to_py_literal(plans_data),
        obs_length_var_literal=to_py_literal(obs_length_var_data),
    )
