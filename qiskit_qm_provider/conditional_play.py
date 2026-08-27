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

"""Provider instruction for a QuAM pulse conditioned by a Qiskit expression.

``ConditionalPlay`` is an :class:`~qiskit.circuit.Instruction`, not a
:class:`~qiskit.circuit.Gate`.  Its sole instruction argument is a typed
Qiskit classical expression, never a numeric Qiskit ``Parameter``.  The
provider exports it as an implicit OpenQASM ``defcal`` call so the expression
reaches the registered QUA macro as a Boolean condition.
"""

from __future__ import annotations

from hashlib import sha256
import re

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError

__all__ = [
    "ConditionalPlay",
    "conditional_play_operation_name",
]


def conditional_play_operation_name(pulse_name: str) -> str:
    """Return the stable OpenQASM-safe operation name for ``pulse_name``.

    The operation name includes a readable ASCII fragment of the QuAM pulse
    label for exported-QASM diagnostics.  It also includes a digest of the
    original UTF-8 label, so distinct labels that sanitize to the same OpenQASM
    fragment (for example, ``"-y90"`` and ``"_y90"``) remain distinct.  The
    digest is part of the name's stable compile-time identity; it is never a
    run-time pulse-selection parameter.
    """
    if not isinstance(pulse_name, str) or not pulse_name:
        raise ValueError("pulse_name must be a non-empty string")
    readable = re.sub(r"[^A-Za-z0-9_]", "_", pulse_name).strip("_")
    if not readable:
        readable = "pulse"
    if readable[0].isdigit():
        readable = f"pulse_{readable}"
    readable = readable[:48].rstrip("_") or "pulse"
    digest = sha256(pulse_name.encode("utf-8")).hexdigest()[:16]
    return f"qm_conditional_play_{readable}_{digest}"


class ConditionalPlay(Instruction):
    """A one-qubit QuAM pulse play guarded by a Boolean classical expression.

    Args:
        pulse_name: Compile-time QuAM pulse label.  It is resolved per physical
            qubit by :meth:`QMBackend.register_conditional_play`.
        condition: A Boolean :mod:`qiskit.circuit.classical.expr` expression.
        label: Optional circuit-display label.

    ``condition`` is stored as the operation's sole :class:`Instruction`
    argument because Qiskit's implicit-defcal exporter dispatches and validates
    argument arity through :attr:`Instruction.params`.  It is not a Qiskit
    ``Parameter`` or a numeric gate parameter.  The provider exporter emits it
    with Qiskit's typed classical-expression AST builder.
    """

    def __init__(self, pulse_name: str, condition: expr.Expr, label: str | None = None):
        if not isinstance(condition, expr.Expr):
            raise CircuitError("ConditionalPlay condition must be a Qiskit classical Expr")
        if condition.type != types.Bool():
            raise CircuitError("ConditionalPlay condition must have Boolean type")

        self.pulse_name = pulse_name
        super().__init__(conditional_play_operation_name(pulse_name), 1, 0, [condition], label=label)

    @property
    def condition_expr(self) -> expr.Expr:
        """The Boolean expression carried in the typed instruction argument slot."""
        return self.params[0]

    @classmethod
    def target_operation(cls, pulse_name: str) -> Instruction:
        """Create the parameter-free Target operation for ``pulse_name``.

        Target applicability and the QUA macro are independent of the runtime
        Boolean expression.  The concrete circuit instruction carries that
        expression in ``params``; this Target exemplar only identifies the
        per-qubit hardware operation.
        """
        return Instruction(conditional_play_operation_name(pulse_name), 1, 0, [])

    def copy(self, name: str | None = None):
        """Copy provider metadata that :meth:`Instruction.copy` does not know.

        The base copy machinery can preserve the generic argument list but has
        no knowledge of the compile-time ``pulse_name`` used to rebuild this
        provider-specific instruction.  Reconstructing the operation keeps the
        pulse identity and its typed Boolean argument together.
        """
        return type(self)(self.pulse_name, self.condition_expr, label=name if name is not None else self.label)

    def inverse(self, annotated: bool = False):
        """Reject inversion instead of applying the base instruction fallback.

        A conditional analog-pulse play has no provider-independent inverse;
        callers must explicitly register the pulse that represents one.
        """
        raise CircuitError("ConditionalPlay has no generic inverse; register and use a separate pulse instead")

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Reject quantum control; QUA's condition is exclusively classical.

        ``Instruction`` does not define a generic controlled form, but this
        explicit method gives callers a stable, explanatory provider error
        rather than allowing an accidental gate-like conversion.
        """
        raise CircuitError("ConditionalPlay cannot be quantum-controlled")

    def __eq__(self, other):
        """Include pulse identity in equality beyond the base argument comparison.

        Two operations with the same generated name shape and Boolean argument
        are interchangeable only when they designate the same QuAM pulse.
        """
        return (
            isinstance(other, ConditionalPlay)
            and self.pulse_name == other.pulse_name
            and self.condition_expr == other.condition_expr
            and self.label == other.label
        )


def _conditional_play(self: QuantumCircuit, pulse_name: str, condition: expr.Expr, qubit, label: str | None = None):
    """Append :class:`ConditionalPlay` to ``self``.

    This provider convenience method mirrors the additional-gate helpers.  The
    corresponding pulse must be registered on the backend before transpilation.
    """
    return self.append(ConditionalPlay(pulse_name, condition, label=label), [qubit])


QuantumCircuit.conditional_play = _conditional_play


def _conditional_play_macro(qubit, pulse_name: str):
    """Build the QUA macro used by a registered conditional play operation."""
    def qua_macro(condition):
        qubit.get_pulse(pulse_name).play(condition=condition)

    return qua_macro
