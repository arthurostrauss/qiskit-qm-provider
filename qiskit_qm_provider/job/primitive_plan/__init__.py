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

"""Per-PUB execution plans shared by the QM primitives.

A plan owns everything one PUB needs across the job: its input tables, the QUA macros that
declare and load those inputs inside the program, and the host-side push order (used both
for local pushes and for the IQCC sync hook, so the two cannot drift apart).

* :class:`~.base.PrimitivePlan` -- circuit parameters (see :mod:`.pub_parameters`).
* :class:`~.sampler.SamplerPlan` -- adds shots.
* :class:`~.estimator.EstimatorPlan` -- adds measurement grouping, observable codes and their packed
  delivery (one int row per binding), and the metadata used to assemble results.

Author: Arthur Strauss
Date: 2026-02-08
"""

from .base import PrimitivePlan
from .estimator import EstimatorPlan
from .sampler import SamplerPlan

__all__ = ["PrimitivePlan", "SamplerPlan", "EstimatorPlan"]
