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

from typing import Callable

from qiskit.circuit import Bit, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.qasm3 import DefcalInstruction, ast
from qiskit.qasm3.exceptions import QASM3ExporterError
from qiskit.qasm3.exporter import Exporter, QASM3Builder
from qiskit.qasm3.printer import BasicPrinter

from .conditional_play import ConditionalPlay

ConditionalPlayValidator = Callable[[object, tuple[int, ...]], None]


class _QMOpenQASM3Builder(QASM3Builder):
    """Qiskit's builder with typed implicit-defcal arguments for ``ConditionalPlay``."""

    _conditional_play_validator: ConditionalPlayValidator | None = None

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
        """Serialize ``ConditionalPlay``'s Boolean argument as a typed expression.

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
        if not isinstance(operation, ConditionalPlay):
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

        qargs = tuple(self.scope.circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if self._conditional_play_validator is not None:
            self._conditional_play_validator(operation, qargs)

        qubits = [self._lookup_bit(qubit) for qubit in instruction.qubits]
        return ast.DefcalCallStatement(
            ident=ast.Identifier(defcal.name),
            parameters=[self.build_expression(operation.condition_expr)],
            qubits=qubits,
            lvalue=None,
        )


class QMOpenQASM3Exporter(Exporter):
    """Qiskit exporter extended with provider implicit defcals."""

    def __init__(self, *args, conditional_play_validator: ConditionalPlayValidator | None = None, **kwargs):
        """Add optional backend preflight validation to Qiskit's exporter setup.

        The base exporter accepts a static ``implicit_defcals`` mapping but has
        no provider validation hook.  This override stores a narrow callback;
        all normal exporter options and semantics remain owned by the base
        class.
        """
        super().__init__(*args, **kwargs)
        self._conditional_play_validator = conditional_play_validator

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
        """
        implicit_defcals = dict(self.implicit_defcals)
        pending = [circuit]
        while pending:
            current = pending.pop()
            for instruction in current.data:
                operation = instruction.operation
                if isinstance(operation, ConditionalPlay):
                    implicit_defcals.setdefault(
                        operation.name,
                        DefcalInstruction(operation.name, parameters=1, qubits=1, return_type=None),
                    )
                pending.extend(getattr(operation, "blocks", ()))
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
        builder._conditional_play_validator = self._conditional_play_validator
        BasicPrinter(stream, indent=self.indent, experimental=self.experimental).visit(builder.build_program())
