import warnings

from quam.components import Qubit
from quam_libs.components import QuAM as Quam, Transmon
from .qm_backend import QMBackend
from typing import Optional, List, Dict, Union
from qm import QuantumMachinesManager
from oqc import QubitsMapping


class FluxTunableTransmonBackend(QMBackend):

    def __init__(
        self,
        machine: Quam,
        qmm: Optional[QuantumMachinesManager] = None,
    ):
        """
        Initialize the QM backend for the Flux-Tunable Transmon based QuAM

        Args:
            machine: The QuAM instance
            channel_mapping: Optional mapping of Qiskit Pulse Channels to QuAM Channels.
                             This mapping enables the conversion of Qiskit schedules into parametric QUA macros.
        """
        if not hasattr(machine, "qubits") or not hasattr(machine, "qubit_pairs"):
            raise ValueError(
                "Invalid QuAM instance provided, should have qubits and qubit_pairs attributes"
            )
        try:
            from qiskit.pulse import DriveChannel, MeasureChannel, ControlChannel
            from quam_qiskit_pulse import FluxChannel

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
