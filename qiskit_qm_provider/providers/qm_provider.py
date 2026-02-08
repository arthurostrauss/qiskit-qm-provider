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

"""QMProvider: load QuAM state and create QMBackend for local/simulator use.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from ..backend.qm_backend import QMBackend
from ..backend.flux_tunable_transmon_backend import FluxTunableTransmonBackend

if TYPE_CHECKING:
    from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam as Quam
    from quam.core import QuamRoot

class QMProvider:
    """
    QMProvider class for Quantum Machines.
    """
    def __init__(self, state_folder_path: Optional[str] = None):
        self.state_folder_path = state_folder_path

    def get_machine(self) -> Quam:
        """
        Get a the latest Quam state from the QMProvider.
        """
        from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam as Quam
        return Quam.load(self.state_folder_path)

    def get_backend(self, machine: Optional[QuamRoot] = None, **backend_options) -> QMBackend:
        """
        Get a QMBackend from the QMProvider.
        """
        from quam.core import QuamRoot
        if machine is not None and not isinstance(machine, QuamRoot):
            raise ValueError("Machine should be a Quam instance")
        return FluxTunableTransmonBackend(machine if machine is not None else self.get_machine(), **backend_options, provider=self)