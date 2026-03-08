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

Requires the ``qm-saas`` optional dependency group::

    pip install qiskit-qm-provider[qm_saas]

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Type

from ..backend import QMBackend

if TYPE_CHECKING:
    from qm import SimulationConfig
    from qm_saas import QmSaas, QOPVersion, QmSaasInstance
    from quam.core import QuamRoot


class QmSaasProvider:
    """Provider for the Quantum Machines SaaS simulation platform.

    Connects to the QM SaaS cloud to simulate QUA programs.  Like
    :class:`QMProvider`, users should supply their own ``quam_cls`` and
    ``backend_cls`` to match their hardware.

    Requires the ``qm_saas`` extras (``pip install qiskit-qm-provider[qm_saas]``).

    Args:
        email: QM SaaS account email.
        password: QM SaaS account password.
        host: QM SaaS platform host URL.
        version: QOP version string (defaults to the latest available).
    """

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

    def get_machine(
        self,
        quam_state_folder_path: Optional[str] = None,
        quam_cls: Type[QuamRoot] | None = None,
    ) -> QuamRoot:
        """Load a QuAM machine.

        Args:
            quam_state_folder_path: Path to the QuAM state folder.
            quam_cls: :class:`~quam.core.QuamRoot` subclass.
                Falls back to ``FluxTunableQuam`` from *quam-builder* when
                omitted, but users are encouraged to provide their own.

        Returns:
            The loaded QuAM machine instance.
        """
        if quam_cls is None:
            from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
                FluxTunableQuam,
            )

            quam_cls = FluxTunableQuam

        if quam_state_folder_path is not None:
            return quam_cls.load(quam_state_folder_path)
        else:
            return quam_cls.load()

    def get_backend(
        self,
        quam_state_folder_path: Optional[str] = None,
        simulation_config: Optional[SimulationConfig] = None,
        quam_cls: Type[QuamRoot] | None = None,
        backend_cls: Type[QMBackend] | None = None,
    ) -> QMBackend:
        """Create a backend connected to a SaaS simulator instance.

        Args:
            quam_state_folder_path: Path to the QuAM state folder.
            simulation_config: Simulation configuration (defaults to 10 000
                clock cycles).
            quam_cls: :class:`~quam.core.QuamRoot` subclass for the machine.
            backend_cls: :class:`QMBackend` subclass to instantiate.
                Defaults to the base :class:`QMBackend`; pass e.g.
                ``FluxTunableTransmonBackend`` or a custom subclass to match
                your hardware.

        Returns:
            A :class:`QMBackend` instance connected to the SaaS simulator.
        """
        from qm import QuantumMachinesManager, SimulationConfig

        machine = self.get_machine(quam_state_folder_path, quam_cls)
        self.instance.spawn()
        qmm = QuantumMachinesManager(
            host=self.instance.host,
            port=self.instance.port,
            connection_headers=self.instance.default_connection_headers,
        )
        if simulation_config is None:
            simulation_config = SimulationConfig(duration=10000)
        if backend_cls is None:
            backend_cls = QMBackend
        return backend_cls(machine, provider=self, qmm=qmm, simulate=simulation_config)

    @property
    def client(self) -> QmSaas:
        """The underlying :class:`QmSaas` client."""
        return self._client

    @property
    def version(self) -> QOPVersion:
        """The QOP version used by this provider."""
        return self._version

    @property
    def instance(self) -> QmSaasInstance:
        """The active SaaS simulator instance."""
        return self._instance

    def close_all(self):
        """Close all SaaS instances and QuantumMachinesManager connections."""
        self._client.close_all()

    def spawn(self):
        """Spawn a new SaaS simulator instance."""
        self._instance.spawn()
