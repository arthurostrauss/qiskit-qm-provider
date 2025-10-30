from __future__ import annotations

from typing import Callable

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
