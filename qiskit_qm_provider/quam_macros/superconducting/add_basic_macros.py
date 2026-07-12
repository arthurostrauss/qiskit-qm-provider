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

"""Seed a QuAM machine with standard superconducting gate-level macros.

Author: Arthur Strauss
Date: 2026-07-12
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from quam.core import QuamRoot

if TYPE_CHECKING:
    from qiskit_qm_provider.backend.qm_backend import QMBackend


def add_basic_macros(
    backend: QuamRoot | QMBackend,
    reset_type: Literal["active", "thermalize"] = "thermalize",
    **reset_macro_kwargs,
):
    """Populate a QuAM machine with standard gate-level macros.

    Adds ``x``, ``sx``, ``rz``, ``sy``, ``sydg``, ``measure``, ``reset``, ``delay``,
    ``id``, and ``cz`` macros. These definitions are **tailored to flux-tunable
    transmon** hardware and assume pulse naming from ``FluxTunableQuam`` /
    quam-builder (e.g. ``x180``, ``x90``, readout pulses, ``CZGate`` on pairs).
    One can either pass a BaseQuam instance or a QMBackend instance.
    If the latter is passed, the target will be updated accordingly.
    Warning: This macro assumes that no other macros are already defined for the qubits.

    This is a convenience starting point, not a universal hardware definition.
    Override macros on your own ``QuamRoot`` for other platforms; coordinate with
    the Quantum Machines team for quam-builder extensions as needed.

    Args:
        backend: A :class:`~.QMBackend` or :class:`~quam.core.QuamRoot` instance.
        reset_type: Reset macro variant, ``"active"`` or ``"thermalize"``.
        **reset_macro_kwargs: Keyword arguments for the reset macro.
    """
    try:
        from quam_builder.architecture.superconducting.custom_gates.single_qubit_gates import (
            ResetMacro,
            VirtualZMacro,
            MeasureMacro,
            DelayMacro,
            IdMacro,
        )
    except ImportError:
        warnings.warn("Could not import single qubit macros from quam_builder. Using macros from qiskit_qm_provider.quam_macros.superconducting.single_qubit_macros. Please upgrade quam-builder to the latest version.")
        from qiskit_qm_provider.quam_macros.superconducting.single_qubit_macros import (
            ResetMacro,
            VirtualZMacro,
            MeasureMacro,
            DelayMacro,
            IdMacro,
        )
    from quam.components.macro import PulseMacro
    from quam_builder.architecture.superconducting.custom_gates.flux_tunable_transmon_pair.two_qubit_gates import (
        CZGate,
    )
    from qiskit_qm_provider.backend.qm_backend import QMBackend

    if not isinstance(backend, (QuamRoot, QMBackend)):
        raise ValueError("Backend should be a QuamRoot or QMBackend instance")
    machine = backend.machine if isinstance(backend, QMBackend) else backend
    pulse_gate_map = {
        "x": "x180",
        "sx": "x90",
        "sy": "y90",
        "sydg": "-y90",
    }
    for qubit in machine.active_qubits:
        if not qubit.macros:
            for gate, pulse in pulse_gate_map.items():
                qubit.macros[gate] = PulseMacro(pulse=pulse)
            qubit.macros["rz"] = VirtualZMacro()
            qubit.macros["measure"] = MeasureMacro(pulse="readout")
            qubit.macros["reset"] = ResetMacro(reset_type=reset_type, pi_pulse="x180", readout_pulse="readout", **reset_macro_kwargs)
            qubit.macros["delay"] = DelayMacro()
            qubit.macros["id"] = IdMacro()

    for qubit_pair in machine.active_qubit_pairs:
        if "cz" not in qubit_pair.macros:
            try:
                qubit_pair.macros["cz"] = None
                qubit_pair.macros["cz"] = CZGate(
                    flux_pulse_control=qubit_pair.qubit_control.z.operations["const"].get_reference(),
                )
            except ValueError as e:
                warnings.warn(f"Could not add default two qubit gates. Add it manually if necessary. Error: {e}")
    if isinstance(backend, QMBackend):
        backend.update_target()
