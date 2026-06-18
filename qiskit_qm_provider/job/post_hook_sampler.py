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

"""Sync-hook generation for the sampler.

The emitted hook depends only on ``iqcc_cloud_client.runtime.get_qm_job`` — it does
not import ``qiskit_qm_provider`` or ``numpy``. Parameter tables are reduced to plain
data here and the push logic is rendered from a Jinja template.

Author: Arthur Strauss
Date: 2026-02-08
"""

from typing import List

from qiskit.primitives.containers.sampler_pub import SamplerPub

from qiskit_qm_provider.parameter_table import ParameterTable
from ._sync_hook_common import jinja_env, serialize_table, to_py_literal


def generate_sync_hook_sampler(pubs: List[SamplerPub], parameter_tables: List[ParameterTable]) -> str:
    """Generate the sync hook code for the sampler."""
    pubs_data = []
    for pub, parameter_table in zip(pubs, parameter_tables):
        table = serialize_table(parameter_table)
        if table is None:
            pubs_data.append(None)
            continue
        pubs_data.append(
            {
                "input_type": table["input_type"],
                "params": table["params"],
                "values": pub.parameter_values.ravel().as_array().tolist(),
            }
        )

    template = jinja_env().get_template("sync_hook_sampler.py.jinja")
    return template.render(pubs_literal=to_py_literal(pubs_data))
