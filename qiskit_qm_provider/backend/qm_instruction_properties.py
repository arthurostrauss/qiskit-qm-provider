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
    """Qiskit instruction properties enriched with a QUA pulse macro.

    ``qua_pulse_macro`` accepts two forms:

    * A :class:`~quam.core.macro.QuamMacro` instance -- a structured QuAM
      component whose attributes (``duration``, ``fidelity``, ``pulse``, …)
      are declared following the `QuAM documentation
      <https://qua-platform.github.io/quam/>`_.  When this form is used,
      ``duration`` and ``error`` are automatically inferred from the macro's
      attributes unless explicitly provided.
    * A plain **callable** -- a bare QUA macro function with no QuAM
      attributes.  In this case ``duration`` and ``error`` must be supplied
      explicitly if they are needed.

    The two forms are transparently unified through the :pyattr:`qua_pulse_macro`
    and :pyattr:`quam_macro` properties.
    """

    def __new__(
        cls,
        duration=None,
        error=None,
        qua_pulse_macro=None,
        quam_macro=None,
        *args,
        **kwargs,
    ):
        macro = quam_macro if quam_macro is not None else qua_pulse_macro
        if macro is not None:
            if duration is None:
                duration = cls._infer_duration(macro)
            if error is None:
                fidelity = getattr(macro, "fidelity", None)
                if isinstance(fidelity, float):
                    error = 1.0 - fidelity

        self = super().__new__(cls, duration=duration, error=error, *args, **kwargs)
        self._qua_pulse_macro = macro
        return self

    @staticmethod
    def _infer_duration(macro) -> float | None:
        """Try to derive a duration (in seconds) from macro attributes.

        Resolution order:
        1. ``macro.duration`` -- used directly if it is numeric.
        2. If ``macro.duration`` is a QuAM string reference (e.g.
           ``"#./inferred_duration"``), resolve via ``macro.inferred_duration``.
        3. Fall back to ``macro.pulse.length * 1e-9`` when available.
        """
        duration = getattr(macro, "duration", None)

        if isinstance(duration, str):
            try:
                duration = macro.inferred_duration
            except (ValueError, AttributeError):
                duration = None

        if duration is None:
            pulse = getattr(macro, "pulse", None)
            if pulse is not None:
                if isinstance(pulse, str):
                    pulse = macro.qubit.get_pulse(pulse)
                duration = pulse.length * 1e-9

        return duration

    def __init__(
        self,
        duration: float | None = None,
        error: float | None = None,
        qua_pulse_macro: Callable | QuamMacro | None = None,
        quam_macro: QuamMacro | None = None,
    ):
        """Create a new ``QMInstructionProperties``.

        Args:
            duration: Gate duration in seconds.  Inferred from ``qua_pulse_macro``
                when a :class:`~quam.core.macro.QuamMacro` is provided and this
                is left as ``None``.
            error: Gate error rate.  Inferred as ``1 - fidelity`` from a
                :class:`~quam.core.macro.QuamMacro` when left as ``None``.
            qua_pulse_macro: Either a :class:`~quam.core.macro.QuamMacro`
                (structured QuAM component with declarative attributes) or a
                plain callable (bare QUA macro function).
            quam_macro: Explicit :class:`~quam.core.macro.QuamMacro` input.
                When both ``quam_macro`` and ``qua_pulse_macro`` are supplied,
                ``quam_macro`` takes priority and ``qua_pulse_macro`` is ignored.
        """
        super().__init__()

    @property
    def qua_pulse_macro(self) -> Callable | None:
        """The QUA macro as a callable.

        If the underlying object is a :class:`~quam.core.macro.QuamMacro`,
        its ``.apply`` method is returned so that callers always receive a
        uniform callable interface.  For a plain callable, it is returned
        as-is.
        """
        if isinstance(self._qua_pulse_macro, QuamMacro):
            return self._qua_pulse_macro.apply
        return self._qua_pulse_macro

    @qua_pulse_macro.setter
    def qua_pulse_macro(self, value: Callable | QuamMacro | None):
        self._qua_pulse_macro = value

    @property
    def quam_macro(self) -> QuamMacro | None:
        """The underlying :class:`~quam.core.macro.QuamMacro`, or ``None``.

        Returns the original :class:`~quam.core.macro.QuamMacro` instance when
        one was provided, giving access to its full attribute set (``duration``,
        ``fidelity``, ``pulse``, etc.).  Returns ``None`` when the macro was
        supplied as a plain callable.
        """
        if isinstance(self._qua_pulse_macro, QuamMacro):
            return self._qua_pulse_macro
        return None

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
