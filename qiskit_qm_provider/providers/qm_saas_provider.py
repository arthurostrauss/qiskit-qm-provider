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

"""QM SaaS provider: connect to Quantum Machines cloud (QmSaas) and get backends.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from ..backend import QMBackend, FluxTunableTransmonBackend

if TYPE_CHECKING:
    from qm import SimulationConfig
    from qm_saas import QmSaas, QOPVersion, QmSaasInstance
    from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
        FluxTunableQuam as Quam,
    )


class QmSaasProvider:
    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        version: Optional[str] = None,
    ):
        from qm_saas import QmSaas, QOPVersion

        if email is None or password is None or host is None:
            import json

            try:
                path = Path.home() / "qm_saas_config.json"
                with open(path, "r") as f:
                    config = json.load(f)
                email = config["email"]
                password = config["password"]
                host = config["host"]
            except FileNotFoundError:
                raise FileNotFoundError(
                    "QM Saas config file not found. Please provide email, password, and host."
                )
        self.email = email
        self.password = password
        self.host = host
        self._client = QmSaas(email=email, password=password, host=host)
        self._version = (
            QOPVersion(version) if version is not None else self.client.latest_version()
        )
        self._instance = self._client.simulator(version=self.version)

    def get_machine(self, quam_state_folder_path: Optional[str] = None) -> Quam:
        """
        Get a Quam instance from the QmSaasProvider.
        """
        from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
            FluxTunableQuam as Quam,
        )

        if quam_state_folder_path is not None:
            return Quam.load(quam_state_folder_path)
        else:
            return Quam.load()

    def get_backend(
        self,
        quam_state_folder_path: Optional[str] = None,
        simulation_config: Optional[SimulationConfig] = None,
    ) -> QMBackend:
        """
        Get a QMBackend from the QmSaasProvider.
        """
        from qm import QuantumMachinesManager, SimulationConfig

        machine = self.get_machine(quam_state_folder_path)
        self.instance.spawn()
        qmm = QuantumMachinesManager(
            host=self.instance.host,
            port=self.instance.port,
            connection_headers=self.instance.default_connection_headers,
        )
        if simulation_config is None:
            simulation_config = SimulationConfig(duration=10000)
        return FluxTunableTransmonBackend(
            machine, provider=self, qmm=qmm, simulate=simulation_config
        )

    @property
    def client(self) -> QmSaas:
        """
        Get the QmSaas client.
        """
        return self._client

    @property
    def version(self) -> QOPVersion:
        """
        Get the QmSaas version.
        """
        return self._version

    @property
    def instance(self) -> QmSaasInstance:
        """
        Get the QmSaas instance.
        """
        return self._instance

    def close_all(self):
        """
        Close all QmSaas instances and QuantumMachinesManager instances.
        """
        self._client.close_all()

    def spawn(self):
        """
        Spawn a new QmSaas instance.
        """
        self._instance.spawn()
