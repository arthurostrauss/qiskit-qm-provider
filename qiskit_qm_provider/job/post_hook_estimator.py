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
not import ``qiskit_qm_provider`` or ``numpy``. Each plan reduces its pushes
(parameter rows and packed observable rows) to plain data and the push loop is
rendered from a Jinja template.

Author: Arthur Strauss
Date: 2026-02-08
"""

from typing import List

from ._sync_hook_common import render_sync_hook
from .primitive_plans import EstimatorPlan


def generate_sync_hook_estimator(plans: List[EstimatorPlan]) -> str:
    """Generate the sync hook code pushing the estimator plans' streamed inputs."""
    return render_sync_hook(plans)
