from __future__ import annotations
import warnings

from .qm_backend import QMBackend
from typing import Optional, List, Union, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from oqc import QubitsMapping
    from quam_libs.components import QuAM as Quam, Transmon, TransmonPair
    from qm import QuantumMachinesManager


class FluxTunableTransmonBackend(QMBackend):

    def __init__(
        self,
        machine: Quam,
        qmm: Optional[QuantumMachinesManager] = None,
        name: Optional[str] = None,
        **fields,
    ):
        """
        Initialize the QM backend for the Flux-Tunable Transmon based QuAM

        Args:
            machine: The QuAM instance
            qmm: A QuantumMachinesManager instance (useful if using cloud simulator or IQCC Cloud)
            name: Name of the backend
            fields: Optional kwargs to specify backend options

        """
        if not hasattr(machine, "qubits") or not hasattr(machine, "qubit_pairs"):
            raise ValueError(
                "Invalid QuAM instance provided, should have qubits and qubit_pairs attributes"
            )
        try:
            from qiskit.pulse import DriveChannel, MeasureChannel, ControlChannel
            from ..pulse.quam_qiskit_pulse import FluxChannel

            drive_channel_mapping = {
                DriveChannel(i): qubit.xy for i, qubit in enumerate(machine.active_qubits)
            }
            flux_channel_mapping = {
                FluxChannel(i): qubit.z for i, qubit in enumerate(machine.active_qubits)
            }
            readout_channel_mapping = {
                MeasureChannel(i): qubit.resonator for i, qubit in enumerate(machine.active_qubits)
            }
            control_channel_mapping = {
                ControlChannel(i): qubit_pair.coupler
                for i, qubit_pair in enumerate(machine.active_qubit_pairs)
            }
            channel_mapping = {
                **drive_channel_mapping,
                **flux_channel_mapping,
                **control_channel_mapping,
                **readout_channel_mapping,
            }
        except ImportError:
            warnings.warn("qiskit.pulse is not available, channel mapping will not be set.")
            channel_mapping = {}
        super().__init__(
            machine,
            channel_mapping=channel_mapping,
            init_macro=machine.apply_all_flux_to_joint_idle,
            qmm=qmm,
            name=name if name is not None else "FluxTunableTransmonBackend",
            **fields,
        )

    @property
    def qubit_mapping(self) -> QubitsMapping:
        """
        Retrieve the qubit to quantum elements mapping for the backend.
        """
        return {
            i: (qubit.xy.name, qubit.z.name, qubit.resonator.name)
            for i, qubit in enumerate(self.machine.active_qubits)
        }

    @property
    def meas_map(self) -> List[List[int]]:
        """
        Retrieve the measurement map for the backend.
        """
        return [[i] for i in range(len(self.machine.active_qubits))]

    def flux_channel(self, qubit: int):
        """
        Retrieve the flux channel for the given qubit.
        """
        try:
            from .quam_qiskit_pulse import FluxChannel

            return FluxChannel(qubit)
        except ImportError:
            raise ImportError("Qiskit Pulse is not available, cannot retrieve flux channel.")

    @property
    def qubits(self) -> List[Transmon]:
        """
        Retrieve the list of Transmon qubits in the backend.
        """
        return super().qubits

    def get_qubit(self, qubit: Union[int, str]) -> Transmon:
        """
        Retrieve a Transmon qubit by its index or name.

        Args:
            qubit: The index or name of the qubit to retrieve.
        Returns:
            Transmon: The Transmon qubit object.
        """
        return super().get_qubit(qubit)

    def get_qubit_pair(self, qubits: Tuple[int|str|Transmon, int|str|Transmon]) -> TransmonPair:
        """
        Retrieve a Transmon pair by its indices or names.

        Args:
            qubits: A tuple containing the indices or names of the qubits in the pair.
        Returns:
            QubitPair: The Transmon pair object.
        """
        return super().get_qubit_pair(qubits)