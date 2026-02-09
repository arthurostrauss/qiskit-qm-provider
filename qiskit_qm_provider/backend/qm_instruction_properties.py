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

"""QMInstructionProperties: Qiskit instruction properties from QuAM pulse macros.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from typing import Callable
from copy import deepcopy

from qiskit.transpiler import InstructionProperties
from quam.core.macro import QuamMacro


class QMInstructionProperties(InstructionProperties):
    def __new__(cls, duration=None, error=None, qua_pulse_macro=None, *args, **kwargs):
        if duration is None and hasattr(qua_pulse_macro, "duration"):
            duration = qua_pulse_macro.duration
            if duration is None and hasattr(qua_pulse_macro, "pulse"):
                if isinstance(qua_pulse_macro.pulse, str):
                    pulse = qua_pulse_macro.qubit.get_pulse(qua_pulse_macro.pulse)
                else:
                    pulse = qua_pulse_macro.pulse
                duration = pulse.length * 1e-9  # Convert to seconds
        if error is None and hasattr(qua_pulse_macro, "fidelity") and isinstance(qua_pulse_macro.fidelity, float):
            error = 1 - qua_pulse_macro.fidelity
        if duration == "#./inferred_duration":
            try:
                duration = qua_pulse_macro.inferred_duration
            except ValueError:
                duration = None
                
        self = super().__new__(cls, duration=duration, error=error, *args, **kwargs)
        self._qua_pulse_macro = qua_pulse_macro
        return self

    def __init__(
        self,
        duration: float | None = None,
        error: float | None = None,
        qua_pulse_macro: Callable | QuamMacro | None = None,
    ):
        super().__init__()

    @property
    def qua_pulse_macro(self) -> Callable | None:
        return self._qua_pulse_macro.apply if isinstance(self._qua_pulse_macro, QuamMacro) else self._qua_pulse_macro

    @qua_pulse_macro.setter
    def qua_pulse_macro(self, value: Callable | QuamMacro | None):
        self._qua_pulse_macro = value

    @property
    def quam_macro(self) -> QuamMacro | None:
        return self._qua_pulse_macro if isinstance(self._qua_pulse_macro, QuamMacro) else None

    @quam_macro.setter
    def quam_macro(self, value: QuamMacro | None):
        self._qua_pulse_macro = value

    def __repr__(self):
        return (
            f"QMInstructionProperties(duration={self.duration}, "
            f"error={self.error}, "
            f"qua_pulse_macro={self._qua_pulse_macro})"
        )

    def __getstate__(self):
        return (super().__getstate__(), self._qua_pulse_macro)

    def __setstate__(self, state: tuple):
        super().__setstate__(state[0])
        self._qua_pulse_macro = state[1]

    def __deepcopy__(self, memo):
        """
        Custom deepcopy that mirrors Qiskit's InstructionProperties semantics
        while keeping the same reference to the underlying macro.

        - The numeric fields (duration, error) are copied like a normal
          InstructionProperties instance.
        - The `_qua_pulse_macro` / `qua_pulse_macro` attribute is **not**
          deep-copied; the reference is shared between copies. This avoids
          attempting to deepcopy or pickle potentially non-picklable QuAM
          macros, while still allowing Qiskit to deepcopy Targets and their
          instruction-property maps.
        """
        if id(self) in memo:
            return memo[id(self)]

        # Recreate via __new__ so that the Rust-side base state is initialized correctly,
        # but pass through the same macro object by reference.
        cls = self.__class__
        result = cls.__new__(
            cls,
            duration=deepcopy(self.duration, memo),
            error=deepcopy(self.error, memo),
            qua_pulse_macro=self._qua_pulse_macro,
        )
        memo[id(self)] = result
        return result
