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

"""OpenQASM 3 export support for provider-specific instructions."""

from __future__ import annotations

from qiskit.circuit import Bit, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.qasm3 import DefcalInstruction, ast
from qiskit.qasm3.exceptions import QASM3ExporterError
from qiskit.qasm3.exporter import Exporter, QASM3Builder
from qiskit.qasm3.printer import BasicPrinter

from .conditional_play import (
    _box_matches_conditional_play_contract,
    _ConditionalPlayInstruction,
    _iter_conditional_plays,
)


class _QMOpenQASM3Builder(QASM3Builder):
    """Qiskit's builder with typed implicit-defcal arguments for a conditional play."""

    def build_box(self, instruction):
        """Build a ``box``, enforcing the conditional-play box contract.

        A box built by ``QuantumCircuit.conditional_play``/``add_conditional_play``
        always holds exactly one ``_ConditionalPlayInstruction`` and nothing
        else (see :mod:`qiskit_qm_provider.conditional_play`). This is the
        last point in the pipeline that still sees the box before it becomes
        hardware-bound OpenQASM 3, so it is where that invariant is actually
        checked, in case a generic transpiler pass altered the box's contents
        in between without knowing to leave it alone. An ordinary box with no
        conditional play in it at all is left to the base builder unchanged.
        """
        body = instruction.operation.blocks[0]
        has_conditional_play = any(isinstance(i.operation, _ConditionalPlayInstruction) for i in body.data)
        if has_conditional_play and not _box_matches_conditional_play_contract(body):
            raise QASM3ExporterError(
                "A box containing a ConditionalPlay must contain exactly that one instruction "
                "and nothing else -- build it via QuantumCircuit.conditional_play or "
                "add_conditional_play."
            )
        return super().build_box(instruction)

    def _resolve_expression_resource(self, resource):
        """Resolve copied circuit resources by their stable Qiskit names.

        Qiskit's generic transpiler does not know that ``condition_expr`` is
        provider metadata, so it does not remap its referenced classical
        resources while copying a circuit.  The current export scope does know
        the corresponding resources, which lets us recover the standard Qiskit
        copy/transpile case without adding a provider-specific transpiler pass.
        """
        circuit = self.scope.circuit
        if isinstance(resource, Bit):
            if resource in self.scope.bit_map:
                return resource
            register = getattr(resource, "_register", None)
            index = getattr(resource, "_index", None)
            if register is not None and index is not None:
                for candidate in circuit.cregs:
                    if candidate.name == register.name and len(candidate) == len(register):
                        return candidate[index]
            if isinstance(resource, Clbit) and index is not None and index < len(circuit.clbits):
                return circuit.clbits[index]
            return resource
        if isinstance(resource, ClassicalRegister):
            for candidate in circuit.cregs:
                if candidate is resource or (candidate.name == resource.name and len(candidate) == len(resource)):
                    return candidate
        return resource

    def _lookup_bit(self, bit):
        """Resolve copied expression bits before the base builder serializes them.

        This differs from :meth:`QASM3Builder._lookup_bit` only for a bit held
        inside a ``ConditionalPlay`` expression after a generic Qiskit copy or
        transpilation.  Such a bit can refer to an equivalent source register
        rather than the current build scope; the base implementation requires
        the latter.  All ordinary bit lookups are delegated unchanged.
        """
        return super()._lookup_bit(self._resolve_expression_resource(bit))

    def _lookup_variable_for_expression(self, var):
        """Resolve copied expression registers before using Qiskit's base lookup.

        This is the register analogue of :meth:`_lookup_bit`.  It changes no
        naming or expression semantics; it only reconnects provider-held
        expression resources to the active export scope, then delegates to the
        standard Qiskit symbol-table lookup.
        """
        return super()._lookup_variable_for_expression(self._resolve_expression_resource(var))

    def build_defcal_call(self, instruction, defcal):
        """Serialize a conditional play's Boolean argument as a typed expression.

        The base method is retained for every other implicit defcal.  For this
        instruction alone, the base implementation is unsuitable because it
        serializes every argument via Qiskit's legacy numeric
        ``ParameterExpression`` path and applies ``pi_check``.  A Boolean
        :class:`~qiskit.circuit.classical.expr.Expr` must instead be lowered by
        :meth:`build_expression`.  The result is otherwise the standard Qiskit
        ``DefcalCallStatement`` and prints as an ordinary OpenQASM operation
        call, with no emitted defcal body.
        """
        operation = instruction.operation
        if not isinstance(operation, _ConditionalPlayInstruction):
            return super().build_defcal_call(instruction, defcal)

        if (
            defcal.parameters != 1
            or defcal.qubits != 1
            or defcal.return_type is not None
            or len(instruction.params) != 1
            or not isinstance(operation.condition_expr, expr.Expr)
            or operation.condition_expr.type != types.Bool()
        ):
            raise QASM3ExporterError(
                "ConditionalPlay requires an implicit defcal signature of (bool) on one qubit with no return value"
            )

        qubits = [self._lookup_bit(qubit) for qubit in instruction.qubits]
        return ast.DefcalCallStatement(
            ident=ast.Identifier(defcal.name),
            parameters=[self.build_expression(operation.condition_expr)],
            qubits=qubits,
            lvalue=None,
        )


class QMOpenQASM3Exporter(Exporter):
    """Qiskit exporter extended with provider implicit defcals."""

    def dump(self, circuit, stream):
        """Export with implicit defcals for encountered conditional plays.

        This differs from :meth:`Exporter.dump` only by selecting the provider
        builder and adding an implicit-defcal descriptor for each encountered
        ``ConditionalPlay``.  That descriptor makes Qiskit dispatch this
        ordinary :class:`Instruction` through ``build_defcal_call`` rather than
        rejecting non-``Gate`` operations.  Its name is removed from the
        builder's basis-gate list because Qiskit's symbol table treats an
        implicit defcal and a basis gate with the same name as conflicting
        declarations.  Existing user-supplied implicit defcals are preserved.

        ``ConditionalPlay`` is always found inside the ``box`` that
        :func:`~qiskit_qm_provider.conditional_play._conditional_play` wraps
        it in (see the module docstring for why); that box is exported as an
        ordinary OpenQASM 3 ``box`` statement, using Qiskit's own unmodified
        ``build_box`` (beyond checking the box's structural contract, see
        ``_QMOpenQASM3Builder.build_box``) -- the QUA compiler's ``visit_Box``
        already unwraps a box's body transparently, so no compiler-side or
        exporter-side flattening is needed here. This exporter has no
        ``Target`` of its own and does not validate that a ``ConditionalPlay``
        is registered anywhere; that is
        :meth:`QMBackend.quantum_circuit_to_qua`'s responsibility.
        """
        implicit_defcals = dict(self.implicit_defcals)
        for operation, _ in _iter_conditional_plays(circuit):
            implicit_defcals.setdefault(
                operation.name,
                DefcalInstruction(operation.name, parameters=1, qubits=1, return_type=None),
            )
        basis_gates = [gate for gate in self.basis_gates if gate not in implicit_defcals]
        builder = _QMOpenQASM3Builder(
            circuit,
            includeslist=self.includes,
            basis_gates=basis_gates,
            disable_constants=self.disable_constants,
            allow_aliasing=self.allow_aliasing,
            experimental=self.experimental,
            annotation_handlers=self.annotation_handlers,
            implicit_defcals=implicit_defcals,
        )
        BasicPrinter(stream, indent=self.indent, experimental=self.experimental).visit(builder.build_program())
