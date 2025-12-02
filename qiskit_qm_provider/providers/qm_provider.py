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