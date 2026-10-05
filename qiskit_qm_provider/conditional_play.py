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

"""Provider support for a QuAM pulse conditioned by a Qiskit expression.

The instruction itself (``_ConditionalPlay``) is private. Its
Boolean condition lives only in ``Instruction.params``, invisible to Qiskit's
DAG, so a bare instance is vulnerable to being silently reordered relative to
a later write of the same classical ``Var`` by any
``circuit_to_dag``/``dag_to_circuit`` round-trip (i.e. by any ``transpile()``
call). The only sanctioned ways to create one are
:meth:`QuantumCircuit.conditional_play` (installed on every circuit below) and
:func:`add_conditional_play`, both of which always wrap it in a ``box`` so its
condition's ``Var`` becomes a real DAG dependency. The provider exports that
box as an ordinary OpenQASM 3 ``box`` statement; see
:meth:`~qiskit_qm_provider.qasm3_exporter._QMOpenQASM3Builder.build_box` for
where its structural contract -- exactly one ``_ConditionalPlay``
and nothing else -- is enforced.
"""

from __future__ import annotations

from hashlib import sha256
import re

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError

__all__ = [
    "add_conditional_play",
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


class _ConditionalPlay(Instruction):
    """A one-qubit QuAM pulse play guarded by a Boolean classical expression.

    Private: this class is never exported, so a caller cannot construct one
    and ``append`` it directly, bypassing the box that keeps its condition
    ordering-safe (see the module docstring). Use
    :meth:`QuantumCircuit.conditional_play` or :func:`add_conditional_play`.

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
        """Add label comparison on top of the base class's identity/params check.

        Base :meth:`Instruction.__eq__` already compares ``name`` (which encodes
        ``pulse_name``, see :func:`conditional_play_operation_name`) and ``params``
        (``condition_expr`` is the sole entry); ``label`` is the only thing it
        doesn't consider.
        """
        return super().__eq__(other) and self.label == other.label


def _box_matches_conditional_play_contract(body: QuantumCircuit) -> bool:
    """Whether ``body`` is exactly one ``_ConditionalPlay`` and nothing else.

    This is the structural contract every box built by
    :meth:`QuantumCircuit.conditional_play`/:func:`add_conditional_play`
    satisfies by construction. It is enforced at OpenQASM 3 export time (see
    :meth:`~qiskit_qm_provider.qasm3_exporter._QMOpenQASM3Builder.build_box`)
    against boxes that may have been altered in between -- e.g. by a generic
    transpiler pass that iterates over ``BoxOp`` nodes without knowing to
    leave this one's body alone.
    """
    return len(body.data) == 1 and isinstance(body.data[0].operation, _ConditionalPlay)


def _iter_conditional_plays(circuit: QuantumCircuit, qubit_map: dict | None = None):
    """Recursively yield every conditional play in ``circuit`` with its physical qargs.

    A conditional play always lives one level inside a ``box`` (see the
    module docstring), and a caller may itself nest that box inside further
    control flow (e.g. an ``if_test``). This composes qubit indices through
    every such level the same way Qiskit composes them when it runs a block
    natively: a block's own qubits correspond positionally to the qargs of the
    instruction that owns it.
    """
    if qubit_map is None:
        qubit_map = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        operation = instruction.operation
        qargs = tuple(qubit_map[qubit] for qubit in instruction.qubits)
        if isinstance(operation, _ConditionalPlay):
            yield operation, qargs
        for block in getattr(operation, "blocks", ()):
            yield from _iter_conditional_plays(block, dict(zip(block.qubits, qargs)))


def _conditional_play(self: QuantumCircuit, pulse_name: str, condition: expr.Expr, qubit, label: str | None = None):
    """Append a conditional play to ``self``, wrapped in a ``box``.

    This provider convenience method mirrors the additional-gate helpers.  The
    corresponding pulse must be registered on the backend before transpilation.

    See the module docstring for why the ``box`` is there (DAG ordering
    correctness against ``transpile``) and why it is exported as-is rather
    than unwrapped.
    """
    with self.box():
        instruction_set = self.append(_ConditionalPlay(pulse_name, condition, label=label), [qubit])
    return instruction_set


QuantumCircuit.conditional_play = _conditional_play


def add_conditional_play(
    qc: QuantumCircuit, pulse_name: str, condition: expr.Expr, qubit, label: str | None = None
):
    """Append a conditional QuAM pulse play to ``qc``, guarded by ``condition``.

    Function-call equivalent of ``qc.conditional_play(...)`` (the method this
    module installs on every :class:`~qiskit.circuit.QuantumCircuit`, above),
    for callers who would rather use a standalone function than rely on the
    monkey-patched method. Both go through the exact same box-wrapped
    construction -- this just calls the method.
    """
    return qc.conditional_play(pulse_name, condition, qubit, label=label)


def _conditional_play_macro(qubit, pulse_name: str):
    """Build the QUA macro used by a registered conditional play operation."""
    def qua_macro(condition):
        qubit.get_pulse(pulse_name).play(condition=condition)

    return qua_macro
