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

"""Flux-tunable transmon backend: QMBackend built from QuAM (flux-tunable) machine config.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations
import warnings

from .qm_backend import QMBackend, requires_qiskit_pulse
from typing import Iterable, Optional, List, Union, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from qm_qasm import QubitsMapping
    from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
        FluxTunableQuam as Quam,
    )
    from quam_builder.architecture.superconducting.qubit.flux_tunable_transmon import (
        FluxTunableTransmon as Transmon,
    )
    from quam_builder.architecture.superconducting.qubit_pair.flux_tunable_transmon_pair import (
        FluxTunableTransmonPair as TransmonPair,
    )
    from qm import QuantumMachinesManager
    from qiskit.pulse import ControlChannel


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

            drive_channel_mapping = {
                DriveChannel(i): qubit.xy
                for i, qubit in enumerate(machine.active_qubits)
            }
            flux_channel_mapping = {
                ControlChannel(i): qubit.z
                for i, qubit in enumerate(machine.active_qubits)
            }
            readout_channel_mapping = {
                MeasureChannel(i): qubit.resonator
                for i, qubit in enumerate(machine.active_qubits)
            }
            control_channel_mapping = {
                ControlChannel(i + len(machine.active_qubits)): qubit_pair.coupler
                for i, qubit_pair in enumerate(machine.active_qubit_pairs)
                if qubit_pair.coupler is not None
            }

            channel_mapping = {
                **drive_channel_mapping,
                **flux_channel_mapping,
                **control_channel_mapping,
                **readout_channel_mapping,
            }
        except ImportError:
            warnings.warn(
                "qiskit.pulse is not available, channel mapping will not be set."
            )
            channel_mapping = {}
        super().__init__(
            machine,
            channel_mapping=channel_mapping,
            init_macro=machine.initialize_qpu,
            qmm=qmm,
            name=name,
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

    @requires_qiskit_pulse
    def control_channel(self, qubits: Iterable[int]) -> List[ControlChannel]:
        """
        Return the secondary drive channel for the given qubit/qubit pair.

        If the qubits are a qubit pair, return the control channel for the qubit pair (tunable coupler).
        If the qubits are a single qubit, return the control channel for the qubit (flux)

        Args:
            qubits: Tuple or list of qubits of the form
                ``(control_qubit, target_qubit)``.

        Returns:
            List[ControlChannel]: The multi qubit control line.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        qubits = tuple(qubits)
        if len(qubits) > 2 or len(qubits) < 1:
            raise ValueError(
                "Control channel should be defined for a qubit pair or a single qubit."
            )
        if len(qubits) == 2:
            qubit_pair = self.get_qubit_pair(qubits)
            if qubit_pair.coupler is not None:
                return [self.get_pulse_channel(qubit_pair.coupler)]
            else:
                return []  # no control channel for the qubit pair
        elif len(qubits) == 1:
            qubit = self.get_qubit(qubits[0])
            if qubit.z is not None:
                return [self.get_pulse_channel(qubit.z)]
            else:
                return []  # no control channel for the qubit

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

    def get_qubit_pair(
        self, qubits: Tuple[int | str | Transmon, int | str | Transmon]
    ) -> TransmonPair:
        """
        Retrieve a Transmon pair by its indices or names.

        Args:
            qubits: A tuple containing the indices or names of the qubits in the pair.
        Returns:
            QubitPair: The Transmon pair object.
        """
        return super().get_qubit_pair(qubits)
