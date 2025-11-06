import json
import os
from typing import Optional
from .backend.flux_tunable_transmon_backend import FluxTunableTransmonBackend
from iqcc_cloud_client import IQCC_Cloud
from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam as Quam

def get_machine_from_iqcc(backend_name: str, api_token: Optional[str] = None):
    try:
        from iqcc_calibration_tools.quam_config.components import Quam as IQCCQuam
    except ImportError:
        IQCCQuam = Quam
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
    
    def get_machine(self, name: str) -> Quam:
        """
        Get a the latest Quam state from the IQCC Cloud.
        """
        machine, _ = get_machine_from_iqcc(name, self.api_token)
        return machine

    def get_cloud_client(self, name: str) -> IQCC_Cloud:
        """
        Get a the IQCC Cloud client.
        """
        return IQCC_Cloud(quantum_computer_backend=name, api_token=self.api_token)

    
    
    def get_backend(self, name: str|Quam) -> FluxTunableTransmonBackend:
        """
        Get a backend from the IQCC Cloud. For now all backends are assumed to be FluxTunableTransmonBackend.
        """
        if isinstance(name, str):
            machine = self.get_machine(name)
        else:
            machine = name
     
        return FluxTunableTransmonBackend(machine)

    