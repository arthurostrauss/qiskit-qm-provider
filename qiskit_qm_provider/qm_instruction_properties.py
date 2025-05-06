from __future__ import annotations

from typing import Callable

from qiskit.transpiler import InstructionProperties


class QMInstructionProperties(InstructionProperties):
    def __init__(
        self,
        duration: float | None = None,
        error: float | None = None,
        qua_pulse_macro: Callable | None = None,
    ):
        super().__init__(duration=duration, error=error)
        self._qua_pulse_macro = qua_pulse_macro

    @property
    def qua_pulse_macro(self) -> Callable | None:
        return self._qua_pulse_macro

    @qua_pulse_macro.setter
    def qua_pulse_macro(self, value: Callable | None):
        self._qua_pulse_macro = value

    def __repr__(self):
        return (
            f"QMInstructionProperties(duration={self.duration}, "
            f"error={self.error}, "
            f"qua_pulse_macro={self.qua_pulse_macro})"
        )

    def __getstate__(self):
        return (super().__getstate__(), self.qua_pulse_macro)

    def __setstate__(self, state: tuple):
        super().__setstate__(state[0])
        self.qua_pulse_macro = state[1]
