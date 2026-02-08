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

"""IQCC cloud provider: get Quantum Machine from IQCC cloud and build FluxTunableTransmonBackend.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations
import json
import os
from typing import Optional, TYPE_CHECKING
from ..backend.flux_tunable_transmon_backend import FluxTunableTransmonBackend

if TYPE_CHECKING:
    from iqcc_cloud_client import IQCC_Cloud
    from iqcc_calibration_tools.quam_config.components import Quam as IQCCQuam

def get_machine_from_iqcc(backend_name: str, api_token: Optional[str] = None):
    from iqcc_cloud_client import IQCC_Cloud
    try:
        from iqcc_calibration_tools.quam_config.components import Quam as IQCCQuam
    except ImportError:
        from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam as IQCCQuam
    iqcc = IQCC_Cloud(quantum_computer_backend=backend_name, api_token=api_token)

    # Get the latest state and wiring files
    latest_wiring = iqcc.state.get_latest("wiring")
    latest_state = iqcc.state.get_latest("state")

    # Get the state folder path from environment variable
    quam_state_folder_path = os.environ["QUAM_STATE_PATH"]

    # Save the files
    with open(os.path.join(quam_state_folder_path, "wiring.json"), "w") as f:
        json.dump(latest_wiring.data, f, indent=4)

    with open(os.path.join(quam_state_folder_path, "state.json"), "w") as f:
        json.dump(latest_state.data, f, indent=4)

    machine = IQCCQuam.load()

    return machine, iqcc
    
class IQCCProvider:
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self._cloud_client = None
    
    def get_machine(self, name: str) -> IQCCQuam:
        """
        Get a the latest Quam state from the IQCC Cloud.
        """
        machine, cloud_client = get_machine_from_iqcc(name, self.api_token)
        self._cloud_client = cloud_client
        return machine

    def get_cloud_client(self, name: str) -> IQCC_Cloud:
        """
        Get a the IQCC Cloud client.
        """
        if self._cloud_client is None or self._cloud_client.backend != name:
            from iqcc_cloud_client import IQCC_Cloud
            self._cloud_client = IQCC_Cloud(quantum_computer_backend=name, api_token=self.api_token)
        return self._cloud_client

    
    def get_backend(self, name: str|IQCCQuam) -> FluxTunableTransmonBackend:
        """
        Get a backend from the IQCC Cloud. For now all backends are assumed to be FluxTunableTransmonBackend.
        """
        if isinstance(name, str):
            machine = self.get_machine(name)
        else:
            machine = name
     
        return FluxTunableTransmonBackend(machine, provider=self)

    