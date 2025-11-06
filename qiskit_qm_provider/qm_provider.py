from typing import Optional, List, Literal
from .backend.qm_backend import QMBackend
from .backend.flux_tunable_transmon_backend import FluxTunableTransmonBackend
from .backend.backend_utils import add_basic_macros_to_machine
from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam as Quam
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
        return Quam.load(self.state_folder_path)

    def get_backend(self, **backend_options) -> QMBackend:
        """
        Get a QMBackend from the QMProvider.
        """
        return FluxTunableTransmonBackend(self.get_machine(), **backend_options, provider=self)