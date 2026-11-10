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

"""Sampler per-PUB execution plan.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...parameter_table import InputType
from ..pub_parameters import build_pub_tables
from .base import PrimitivePlan

if TYPE_CHECKING:
    from qiskit.primitives.containers.sampler_pub import SamplerPub


@dataclass(eq=False)
class SamplerPlan(PrimitivePlan):
    """Plan for one :class:`~qiskit.primitives.containers.sampler_pub.SamplerPub`."""

    @classmethod
    def from_pub(cls, pub: "SamplerPub", input_type: Optional[InputType], index: int) -> "SamplerPlan":
        """Build a plan for ``pub``; ``index`` names its parameter tables uniquely in the job."""
        circuit_table, param_table = build_pub_tables(pub.circuit, input_type, f"param_table_{index}")
        return cls(pub, index, circuit_table, param_table)

    @property
    def shots(self) -> int:
        """Shot count requested by this PUB."""
        return self.pub.shots

    @property
    def has_tasks(self) -> bool:
        """``False`` for a PUB with zero bindings: it is not compiled, sent or run, and its
        result is empty (shape ``pub.shape``), as with Qiskit's reference sampler."""
        return self.num_bindings > 0
