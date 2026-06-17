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

"""User-facing wrapper around qm-qasm compilation results and measurement outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from qm.qua._expressions import QuaArrayVariable
from qm.qua.type_hints import StreamType
from quam.utils.qua_types import QuaVariable, QuaScalarInt

from ..parameter_table import ParameterPool
from ..parameter_table._mixins import QuaFieldTable
from ..parameter_table._scope import require_qua_program
from .backend_utils import _QASM3_DUMP_LOOSE_BIT_PREFIX
from .measurement_field import MeasurementRegisterField

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qm_qasm import CompilationResult


def _sanitize_output_table_name(qc_name: str) -> str:
    base = qc_name if qc_name and qc_name.isidentifier() else "circuit"
    return f"{base}_output"


_used_output_table_names: dict[str, int] = {}


def _allocate_output_table_name(qc_name: str) -> str:
    """Return a unique table name, suffixing ``_2``, ``_3``, … on reuse."""
    base = _sanitize_output_table_name(qc_name)
    count = _used_output_table_names.get(base, 0) + 1
    _used_output_table_names[base] = count
    if count == 1:
        return base
    return f"{base}_{count}"


def reset_output_table_name_registry() -> None:
    """Clear output-table name counters (also invoked from :meth:`ParameterPool.reset`)."""
    _used_output_table_names.clear()


def _loose_bit_keys(qc: "QuantumCircuit") -> list[str]:
    loose_count = len([bit for bit in qc.clbits if len(qc.find_bit(bit).registers) == 0])
    return [f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{i}" for i in range(loose_count)]


class MeasurementOutcomeTable(QuaFieldTable):
    """Local-only grouping of measurement output fields.

    Tracked in :data:`ParameterPool._measurement_outcome_tables`, not in the runtime
    OPNIC registry. Field names may match runtime struct fields or creg names; resolve
    handles via ``comp.outputs.get_parameter(name)`` vs your input ``ParameterTable``.
    """

    _is_measurement_outcome_table = True

    def __init__(
        self,
        fields: dict[str, MeasurementRegisterField],
        name: str,
    ):
        """Group wired measurement fields under a unique table name.

        Args:
            fields: Mapping from program output key to
                :class:`~qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField`.
            name: Human-readable table name (e.g. ``my_circuit_output``).
        """
        self.table: dict[str, MeasurementRegisterField] = fields
        self.name = name
        self._id = 0
        ParameterPool._register_measurement_outcome_table(self)

    @property
    def parameters(self) -> Sequence[MeasurementRegisterField]:
        """All :class:`~qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField` handles in this table."""
        return list(self.table.values())

    @property
    def state_ints(self) -> dict[str, QuaScalarInt]:
        """Bulk accessor: ``{output_key: packed_int_var}`` for every field.

        Must be accessed inside ``with program():``. Equivalent to
        ``{name: self.get_parameter(name).state_int for name in self.table}``.
        """
        require_qua_program("MeasurementOutcomeTable.state_ints")
        return {name: self.get_parameter(name).state_int for name in self.table}

    @property
    def streams(self) -> dict[str, StreamType]:
        """Bulk accessor: ``{output_key: stream}`` for every field.

        Must be accessed inside ``with program():``. Equivalent to
        ``{name: self.get_parameter(name).stream for name in self.table}``.
        """
        require_qua_program("MeasurementOutcomeTable.streams")
        return {name: self.get_parameter(name).stream for name in self.table}

    def declare(
        self, pause_program=False, declare_stream=True
    ) -> QuaVariable | QuaArrayVariable | Sequence[QuaVariable | QuaArrayVariable]:
        """No-op compatibility shim — measurement vars are compiler-owned.

        Returns the wired QUA variable(s) without declaring anything new.
        """
        variables = [parameter.var for parameter in self.parameters]
        if len(variables) == 1:
            return variables[0]
        return variables

    def rewire(
        self,
        qc: QuantumCircuit,
        compilation_result: CompilationResult,
        *,
        parent: QuaCircuitCompilation | None = None,
    ) -> None:
        """Refresh wiring from a new compilation result (same table object).

        Re-binds each field's
        :attr:`~qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField.var`
        from
        ``compilation_result.result_program`` and invalidates cached ``state_int`` /
        ``stream`` handles when size or compilation identity changes.

        Args:
            qc: Source circuit whose cregs and loose clbits define output keys.
            compilation_result: New qm-qasm result to wire from.
            parent: If given, updates ``parent._fields`` to match this table.
        """
        for creg in qc.cregs:
            field = self.table[creg.name]
            field._wire_from_result(compilation_result, creg.name, register_size=creg.size)
        for key in _loose_bit_keys(qc):
            self.table[key]._wire_from_result(compilation_result, key, register_size=1)
        if parent is not None:
            parent._fields.update(self.table)

    @classmethod
    def from_compilation(
        cls,
        qc: QuantumCircuit,
        compilation_result: CompilationResult,
        *,
        parent: QuaCircuitCompilation | None = None,
        compute_state_int: bool = True,
    ) -> MeasurementOutcomeTable:
        """Build a measurement output table from a compiled circuit.

        Creates one
        :class:`~qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField`
        per classical register and
        per loose clbit (``_bit0``, ``_bit1``, …), wires each from
        ``compilation_result.result_program``, and registers the table in
        :data:`~qiskit_qm_provider.parameter_table.ParameterPool._measurement_outcome_tables`.

        Args:
            qc: Circuit whose measurement outcomes should be exposed.
            compilation_result: qm-qasm result with ``result_program[key]`` entries.
            parent: Optional :class:`~qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation` that owns the fields.
            compute_state_int: Passed through to each field's ``state_int`` support.

        Returns:
            A wired :class:`MeasurementOutcomeTable` (not registered in the runtime
            OPNIC pool).
        """
        fields: dict[str, MeasurementRegisterField] = {}
        for creg in qc.cregs:
            field = MeasurementRegisterField(
                creg.name,
                creg.size,
                compute_state_int=compute_state_int,
                parent=parent,
            )
            field._wire_from_result(compilation_result, creg.name, register_size=creg.size)
            fields[creg.name] = field

        for key in _loose_bit_keys(qc):
            field = MeasurementRegisterField(
                key,
                1,
                compute_state_int=compute_state_int,
                parent=parent,
            )
            field._wire_from_result(compilation_result, key, register_size=1)
            fields[key] = field

        if parent is not None:
            parent._fields.update(fields)

        return cls(fields, name=_allocate_output_table_name(qc.name))


class QuaCircuitCompilation:
    """Wrapper around :class:`~qm_qasm.CompilationResult` with ergonomic measurement outputs.

    Returned by :meth:`~qiskit_qm_provider.backend.QMBackend.quantum_circuit_to_qua`.
    Delegates unknown attributes
    to the underlying compilation result and exposes wired measurement handles via
    :attr:`outputs`.
    """

    def __init__(
        self,
        compilation_result: CompilationResult,
        circuit: "QuantumCircuit",
        *,
        compute_state_int: bool = True,
    ):
        """Wrap a compilation result and wire measurement output fields.

        Args:
            compilation_result: Raw qm-qasm compilation result.
            circuit: The Qiskit circuit that was compiled (defines output keys).
            compute_state_int: Whether fields expose lazy :attr:`state_int` packing.
        """
        self._compilation_result = compilation_result
        self._circuit = circuit
        self._compute_state_int = compute_state_int
        self._fields: dict[str, MeasurementRegisterField] = {}
        self._outputs = MeasurementOutcomeTable.from_compilation(
            circuit,
            compilation_result,
            parent=self,
            compute_state_int=compute_state_int,
        )

    @property
    def compilation_result(self) -> CompilationResult:
        """Underlying qm-qasm :class:`~qm_qasm.CompilationResult`."""
        return self._compilation_result

    @property
    def circuit(self) -> QuantumCircuit:
        """The Qiskit circuit that was compiled."""
        return self._circuit

    @property
    def outputs(self) -> MeasurementOutcomeTable:
        """Compilation-local measurement output table (:attr:`state_ints`, :attr:`streams`)."""
        return self._outputs

    @property
    def fields(self) -> dict[str, MeasurementRegisterField]:
        """Compilation-scoped measurement field handles keyed by program output name."""
        return dict(self._fields)

    def rewire_outputs(
        self,
        circuit: QuantumCircuit,
        compilation_result: CompilationResult,
    ) -> None:
        """Re-bind measurement fields from a new compilation (same wrapper object).

        Updates :attr:`circuit`, :attr:`compilation_result`, and delegates to
        :meth:`MeasurementOutcomeTable.rewire`.

        Args:
            circuit: Updated source circuit.
            compilation_result: New qm-qasm result to wire from.
        """
        self._circuit = circuit
        self._compilation_result = compilation_result
        self._outputs.rewire(circuit, compilation_result, parent=self)

    @property
    def qua_program(self):
        """The QUA DSL program object (``result_program.dsl_program``)."""
        return self._compilation_result.result_program.dsl_program

    def __getattr__(self, name: str) -> Any:
        """Delegate missing attributes to the wrapped compilation result."""
        return getattr(self._compilation_result, name)

    def __repr__(self) -> str:
        return f"QuaCircuitCompilation(name={self._compilation_result.name!r}, " f"outputs={self._outputs.name!r})"
