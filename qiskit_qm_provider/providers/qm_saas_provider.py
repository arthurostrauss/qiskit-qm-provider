import json
import os
from pathlib import Path
from typing import Optional
from qm_saas import QmSaas, QOPVersion
from ..backend import QMBackend, FluxTunableTransmonBackend
from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam as Quam
from qm import QuantumMachinesManager, SimulationConfig

class QmSaasProvider:
    def __init__(self, email: Optional[str] = None, password: Optional[str] = None, host: Optional[str] = None, version: Optional[str] = None):
        if email is None or password is None or host is None:
            try:
                path = Path.home() / "qm_saas_config.json"
                with open(path, "r") as f:
                    config = json.load(f)
                email = config["email"]
                password = config["password"]
                host = config["host"]
            except FileNotFoundError:
                raise FileNotFoundError("QM Saas config file not found. Please provide email, password, and host.")
        self.email = email
        self.password = password
        self.host = host
        self.client = QmSaas(email=email, password=password, host=host)
        self.version = QOPVersion(version) if version is not None else self.client.latest_version()
        self.instance = self.client.simulator(version=self.version)

    def get_machine(self, quam_state_folder_path: Optional[str] = None) -> Quam:
        """
        Get a Quam instance from the QmSaasProvider.
        """
        if quam_state_folder_path is not None:
            return Quam.load(quam_state_folder_path)
        else:
            return Quam.load()
    
    def get_backend(self, quam_state_folder_path: Optional[str] = None, simulation_config: Optional[SimulationConfig] = None) -> QMBackend:
        """
        Get a QMBackend from the QmSaasProvider.
        """
        machine = self.get_machine(quam_state_folder_path)
        self.instance.spawn()
        qmm = QuantumMachinesManager(host=self.instance.host, port=self.instance.port, connection_headers=self.instance.default_connection_headers)
        if simulation_config is None:
            simulation_config = SimulationConfig(duration=10000)
        return FluxTunableTransmonBackend(machine, provider=self, qmm=qmm, simulate=simulation_config)
        



    
    