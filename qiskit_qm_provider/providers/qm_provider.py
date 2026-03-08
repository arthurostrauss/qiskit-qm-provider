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

"""QMProvider: load QuAM state and create QMBackend for local/simulator use.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Type
from ..backend.qm_backend import QMBackend

if TYPE_CHECKING:
    from quam.core import QuamRoot as Quam


class QMProvider:
    """Provider for Quantum Machines hardware using a local QuAM state.

    ``QMProvider`` is hardware-agnostic: users supply their own
    :class:`~quam.core.QuamRoot` subclass (via ``quam_cls``) and their own
    :class:`QMBackend` subclass (via ``backend_cls``) to match their specific
    hardware.  This avoids a hard dependency on any particular architecture
    (e.g. flux-tunable transmons) and lets any QuAM-compatible machine be
    used with the Qiskit stack.

    If ``quam_cls`` is omitted the provider falls back to
    ``FluxTunableQuam`` from *quam-builder* for backward compatibility,
    but users are encouraged to provide their own class.

    Example::

        from qiskit_qm_provider import QMProvider
        from my_lab.quam import MyCustomQuam
        from my_lab.backend import MyBackend

        provider = QMProvider(
            state_folder_path="/path/to/quam/state",
            quam_cls=MyCustomQuam,
        )
        backend = provider.get_backend(backend_cls=MyBackend)
    """

    def __init__(
        self,
        state_folder_path: Optional[str] = None,
        quam_cls: Type[Quam] | None = None,
    ):
        self.state_folder_path = state_folder_path
        self._quam_cls: Type[Quam] = quam_cls
        if self._quam_cls is None:
            from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
                FluxTunableQuam as Quam,
            )

            self._quam_cls = Quam

    def get_machine(self) -> Quam:
        """Load and return the latest QuAM state."""
        return self._quam_cls.load(self.state_folder_path)

    def get_backend(
        self,
        machine: Optional[Quam] = None,
        backend_cls: Type[QMBackend] | None = None,
        **backend_options,
    ) -> QMBackend:
        """Create a :class:`QMBackend` (or subclass) from a QuAM machine.

        Users should pass ``backend_cls`` to select the backend implementation
        that matches their hardware.  When omitted the base :class:`QMBackend`
        is used, which provides the full circuit-to-QUA pipeline but no
        hardware-specific channel mapping or initialization macro.

        Args:
            machine: A pre-loaded :class:`~quam.core.QuamRoot` instance.
                If ``None``, the provider loads one via :meth:`get_machine`.
            backend_cls: The :class:`QMBackend` subclass to instantiate.
                Common choices include ``FluxTunableTransmonBackend`` (for
                flux-tunable transmon setups) or a user-defined subclass.
                Defaults to :class:`QMBackend`.
            **backend_options: Forwarded to the backend constructor.  Typical
                keys: ``qmm``, ``name``, ``shots``, ``compiler_options``,
                ``simulate``, ``memory``, ``skip_reset``, ``meas_level``,
                ``meas_return``.

        Returns:
            A :class:`QMBackend` instance (or the requested subclass).
        """
        from quam.core import QuamRoot as Quam

        if machine is not None and not isinstance(machine, Quam):
            raise ValueError("Machine should be a Quam instance")
        if backend_cls is None:
            backend_cls = QMBackend
        return backend_cls(
            machine if machine is not None else self.get_machine(),
            **backend_options,
            provider=self,
        )
