"""Tests for provider-specific conditional QUA pulse plays."""

import re
import warnings
from unittest.mock import Mock, patch
from types import SimpleNamespace

import pytest
from quam.components import Qubit
from qiskit import transpile
from qiskit.circuit import ClassicalRegister, Gate, Instruction, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow.box import BoxOp
from qiskit.circuit.exceptions import CircuitError
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import Target

from qiskit.qasm3.exceptions import QASM3ExporterError

from qiskit_qm_provider import add_conditional_play
from qiskit_qm_provider.backend.backend_utils import ensure_native_operations
from qiskit_qm_provider.backend.qm_backend import QMBackend
from qiskit_qm_provider.conditional_play import (
    _box_matches_conditional_play_contract,
    _ConditionalPlay,
    _conditional_play_macro,
    conditional_play_operation_name,
)
from qiskit_qm_provider.qasm3_exporter import QMOpenQASM3Exporter


def _exporter_for(pulse_name: str):
    return QMOpenQASM3Exporter(
        includes=(),
        basis_gates=[conditional_play_operation_name(pulse_name)],
        disable_constants=True,
    )


def _unwrap_conditional_play(operation: Instruction) -> _ConditionalPlay:
    """Return the conditional play nested inside ``qc.conditional_play``'s box wrapper."""
    assert isinstance(operation, BoxOp)
    (inner,) = operation.body.data
    assert isinstance(inner.operation, _ConditionalPlay)
    return inner.operation


def _backend_with_pulse(*, get_pulse_side_effect=None):
    qubit = Mock(spec=Qubit)
    qubit.name = "q0"
    qubit.T1 = qubit.T2echo = qubit.f_01 = None
    qubit.macros = {}
    qubit.channels = []
    pulse = Mock()
    pulse.length = 32
    if get_pulse_side_effect is None:
        qubit.get_pulse.return_value = pulse
    else:
        qubit.get_pulse.side_effect = get_pulse_side_effect
    machine = SimpleNamespace(
        qubits={},
        qubit_pairs={},
        active_qubits=[qubit],
        active_qubit_pairs=[],
        network={},
    )
    return QMBackend(machine), qubit, pulse


class TestConditionalPlay:
    def test_operation_name_is_readable_qasm_safe_and_injective(self):
        # An already-valid identifier is used as-is -- no mangling, no digest.
        assert conditional_play_operation_name("x180") == "qm_conditional_play_x180"

        # The one other character this domain's pulse names actually use ("-", the
        # negated-pulse convention) gets a dedicated, still digest-free substitution.
        assert conditional_play_operation_name("-x90") == "qm_conditional_play_minus_x90"

        # That substitution must not collide with how "_" already passes through
        # unchanged -- these must stay distinct.
        assert conditional_play_operation_name("-y90") != conditional_play_operation_name("_y90")
        assert conditional_play_operation_name("-y90") != conditional_play_operation_name("y90")

        # A genuinely unanticipated character still falls back to a digest-suffixed,
        # but still OpenQASM-safe, name.
        exotic_name = conditional_play_operation_name("x 180")
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", exotic_name)
        assert exotic_name != conditional_play_operation_name("x_180")

    def test_circuit_method_preserves_expression_and_pulse_name(self):
        qc = QuantumCircuit(1, 1)
        condition = expr.equal(qc.clbits[0], expr.lift(False))

        qc.conditional_play("x180", condition, 0)

        assert isinstance(qc.data[0].operation, BoxOp)
        operation = _unwrap_conditional_play(qc.data[0].operation)
        assert isinstance(operation, _ConditionalPlay)
        assert isinstance(operation, Instruction)
        assert not isinstance(operation, Gate)
        assert operation.pulse_name == "x180"
        assert operation.condition_expr == condition
        assert operation.params == [condition]

    def test_conditional_play_survives_dag_roundtrip_ordering(self):
        """Regression test: a later ``store`` to the play's own condition Var must not be
        reordered ahead of the play by a circuit_to_dag/dag_to_circuit round-trip.

        Before the ``box`` wrap, ``ConditionalPlay``'s condition read was invisible to the
        DAG (stored only in ``params``), so the round-trip could freely move a later
        ``store`` of the same Var ahead of the play.
        """
        qc = QuantumCircuit(1)
        v = qc.add_var("v", expr.lift(False))
        qc.x(0)
        qc.store(v, expr.lift(True))
        qc.conditional_play("x180", v, 0)
        qc.store(v, expr.lift(False))

        roundtripped = dag_to_circuit(circuit_to_dag(qc))

        names = [instruction.operation.name for instruction in roundtripped.data]
        first_store_index = names.index("store")
        box_index = names.index("box")
        last_store_index = len(names) - 1 - names[::-1].index("store")
        assert first_store_index < box_index < last_store_index

        transpiled = transpile(qc, optimization_level=0)
        names = [instruction.operation.name for instruction in transpiled.data]
        first_store_index = names.index("store")
        box_index = names.index("box")
        last_store_index = len(names) - 1 - names[::-1].index("store")
        assert first_store_index < box_index < last_store_index

    def test_boolean_expression_is_emitted_as_gate_argument(self):
        qc = QuantumCircuit(1, 1)
        condition = expr.equal(qc.clbits[0], expr.lift(False))
        qc.conditional_play("x180", condition, 0)

        qasm = _exporter_for("x180").dumps(qc)

        assert f"{conditional_play_operation_name('x180')}(c[0] == false) q[0];" in qasm
        assert "input float" not in qasm
        assert "box {" in qasm, "the box is a real part of the compiled program, not a Qiskit-side-only artifact"

    def test_box_contract_helper_accepts_a_box_built_via_conditional_play(self):
        qc = QuantumCircuit(1, 1)
        condition = expr.equal(qc.clbits[0], expr.lift(False))
        qc.conditional_play("x180", condition, 0)

        box = qc.data[0].operation
        assert _box_matches_conditional_play_contract(box.body)

    def test_box_contract_helper_rejects_extra_content(self):
        """The structural contract a conditional-play box must satisfy: exactly one
        _ConditionalPlay and nothing else. This must reject a body that a pass
        has added content to, e.g. because it didn't know to leave this box's body alone.
        """
        body = QuantumCircuit(1)
        body.append(_ConditionalPlay("x180", expr.lift(True)), [0])
        body.x(0)

        assert not _box_matches_conditional_play_contract(body)

    def test_qasm3_exporter_rejects_a_malformed_conditional_play_box(self):
        """There is no annotation to signal intent anymore (see the module docstring); the
        box's structural contract is checked directly at export time, in build_box, against
        a box a generic transpiler pass may have altered without knowing to leave it alone.
        """
        qc = QuantumCircuit(1, 1)
        condition = expr.equal(qc.clbits[0], expr.lift(False))
        with qc.box():
            qc.append(_ConditionalPlay("x180", condition), [0])
            qc.x(0)

        with pytest.raises(QASM3ExporterError, match="must contain exactly that one instruction"):
            _exporter_for("x180").dumps(qc)

    def test_add_conditional_play_function_matches_the_method(self):
        """add_conditional_play is a standalone-function equivalent of qc.conditional_play,
        for callers who'd rather not rely on the monkey-patched method -- both must produce
        the exact same box-wrapped construction.
        """
        qc = QuantumCircuit(1, 1)
        condition = expr.equal(qc.clbits[0], expr.lift(False))

        add_conditional_play(qc, "x180", condition, 0)

        assert isinstance(qc.data[0].operation, BoxOp)
        operation = _unwrap_conditional_play(qc.data[0].operation)
        assert operation.pulse_name == "x180"
        assert operation.condition_expr == condition

    def test_boolean_input_variable_is_emitted_as_gate_argument(self):
        qc = QuantumCircuit(1)
        flag = qc.add_input("flag", types.Bool())
        qc.conditional_play("x180", expr.logic_not(flag), 0)

        qasm = _exporter_for("x180").dumps(qc)

        assert "input bool flag;" in qasm
        assert f"{conditional_play_operation_name('x180')}(!flag) q[0];" in qasm

    def test_transpile_keeps_exportable_condition_metadata(self):
        qc = QuantumCircuit(1, 1)
        condition = expr.equal(qc.clbits[0], expr.lift(False))
        qc.conditional_play("x180", condition, 0)
        target = Target()
        target.add_instruction(_ConditionalPlay.target_operation("x180"), {(0,): None})
        target.add_instruction(BoxOp, name="box")

        transpiled = transpile(qc, target=target)
        qasm = _exporter_for("x180").dumps(transpiled)

        assert f"{conditional_play_operation_name('x180')}(c[0] == false) $0;" in qasm
        assert "box {" in qasm
        assert _unwrap_conditional_play(transpiled.data[0].operation).pulse_name == "x180"

    def test_compose_onto_a_renamed_register_remaps_the_condition(self):
        """Composing onto a circuit whose classical register has a different name must
        export the condition against the host's register. Qiskit's ``compose`` remaps
        the ``Expr`` held in the instruction's params itself; the exporter relies on
        that rather than re-resolving resources by name."""
        source = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "c"))
        source.measure(0, 0)
        source.conditional_play("x180", expr.lift(source.clbits[0]), 0)
        host = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "d"))
        host.compose(source, inplace=True)

        qasm = _exporter_for("x180").dumps(host)

        assert f"{conditional_play_operation_name('x180')}(d[0]) q[0];" in qasm
        assert "c[0]" not in qasm

    def test_non_boolean_or_non_expression_condition_is_rejected(self):
        qc = QuantumCircuit(1, 1)
        with pytest.raises(CircuitError, match="classical Expr"):
            _ConditionalPlay("x180", False)
        with pytest.raises(CircuitError, match="Boolean type"):
            _ConditionalPlay("x180", expr.lift(1))

    def test_inverse_and_quantum_control_are_rejected(self):
        qc = QuantumCircuit(1, 1)
        operation = _ConditionalPlay("x180", expr.equal(qc.clbits[0], expr.lift(False)))

        with pytest.raises(CircuitError, match="inverse"):
            operation.inverse()
        with pytest.raises(CircuitError, match="quantum-controlled"):
            operation.control()

    def test_copy_follows_the_standard_instruction_rename_contract(self):
        """copy(name=...) must rename the operation like Instruction.copy does for every
        other instruction, not silently overwrite .label instead.
        """
        condition = expr.lift(True)
        operation = _ConditionalPlay("x180", condition, label="mylabel")

        renamed = operation.copy(name="renamed")

        assert renamed.name == "renamed"
        assert renamed.label == "mylabel"
        assert renamed.pulse_name == "x180"
        assert renamed.condition_expr == condition

        plain_copy = operation.copy()
        assert plain_copy.name == operation.name
        assert plain_copy == operation

    def test_qua_macro_resolves_the_quam_pulse_and_forwards_condition(self):
        pulse = Mock()
        qubit = Mock()
        qubit.get_pulse.return_value = pulse
        condition = object()

        _conditional_play_macro(qubit, "x180")(condition)

        qubit.get_pulse.assert_called_once_with("x180")
        pulse.play.assert_called_once_with(condition=condition)


class TestConditionalPlayRegistration:
    def test_qasm3_exporter_does_not_validate_against_a_target(self):
        """The exporter has no Target of its own; an unregistered ConditionalPlay exports
        fine through it. Validation against the backend's Target is
        quantum_circuit_to_qua's responsibility (see the test below), not the exporter's.
        """
        qc = QuantumCircuit(1, 1)
        qc.conditional_play("x180", expr.equal(qc.clbits[0], expr.lift(False)), 0)

        qasm = _exporter_for("x180").dumps(qc)

        assert f"{conditional_play_operation_name('x180')}(c[0] == false) q[0];" in qasm

    def test_quantum_circuit_to_qua_rejects_an_unregistered_conditional_play(self):
        backend, _, _ = _backend_with_pulse()
        qc = QuantumCircuit(1)
        qc.conditional_play("x180", expr.lift(True), 0)
        qc.ensure_physical(backend.num_qubits)

        with pytest.raises(ValueError, match="not registered"):
            backend.quantum_circuit_to_qua(qc)

    def test_registration_populates_target_and_compiler_mapping(self):
        backend, qubit, pulse = _backend_with_pulse()
        backend.register_conditional_play("x180")
        operation_name = conditional_play_operation_name("x180")

        assert operation_name in backend.target.operation_names
        assert backend.target.instruction_supported(operation_name, (0,))
        macro = backend._operation_mapping_QUA[(operation_name, 1, (0,))]
        condition = object()
        macro(condition)
        qubit.get_pulse.assert_called_with("x180")
        pulse.play.assert_called_once_with(condition=condition)

    def test_registering_a_colliding_pulse_name_warns(self):
        """``"-y90"`` and a pulse literally named ``"minus_y90"`` share one operation
        name; the second registration replaces the first, so it must say so."""
        backend, _, _ = _backend_with_pulse()
        backend.register_conditional_play("-y90")

        with pytest.warns(UserWarning, match="same operation name"):
            backend.register_conditional_play("minus_y90")

    def test_re_registering_the_same_pulse_does_not_warn(self):
        backend, _, _ = _backend_with_pulse()
        backend.register_conditional_play("-y90")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            backend.register_conditional_play("-y90")

    def test_failed_colliding_registration_neither_warns_nor_replaces(self):
        """A colliding registration that fails pulse resolution must leave the earlier
        registration in place and must not claim to have replaced it."""
        backend, qubit, pulse = _backend_with_pulse()
        backend.register_conditional_play("-y90")
        operation_name = conditional_play_operation_name("-y90")
        macro_before = backend._operation_mapping_QUA[(operation_name, 1, (0,))]
        qubit.get_pulse.side_effect = ValueError("Pulse minus_y90 not found")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="minus_y90 not found"):
                backend.register_conditional_play("minus_y90")

        assert backend._conditional_play_pulses[operation_name] == "-y90"
        assert backend._operation_mapping_QUA[(operation_name, 1, (0,))] is macro_before

    @pytest.mark.parametrize("error", [ValueError("Pulse x180 not found"), ValueError("Pulse x180 is not unique")])
    def test_registration_propagates_quam_pulse_lookup_errors(self, error):
        backend, _, _ = _backend_with_pulse(get_pulse_side_effect=error)

        with pytest.raises(ValueError, match=str(error)):
            backend.register_conditional_play("x180")

        assert conditional_play_operation_name("x180") not in backend.target.operation_names

    def test_quantum_circuit_to_qua_ensures_the_circuit_is_physical(self):
        """quantum_circuit_to_qua ensures the circuit is physical via ensure_circuit_physical,
        raising a ValueError if it was not transpiled to physical qubits.
        """
        backend, _, _ = _backend_with_pulse()
        backend.register_conditional_play("x180")
        qc = QuantumCircuit(QuantumRegister(1, "myreg"))
        qc.conditional_play("x180", expr.lift(True), 0)

        with pytest.raises(ValueError, match="not expressed in terms of this backend's"):
            backend.quantum_circuit_to_qua(qc)

        assert qc.qregs[0].name == "myreg", "rejecting a non-physical circuit must not mutate it"
        assert qc.layout is None, "rejecting a non-physical circuit must not mutate it"

    def test_native_operation_check_accepts_a_registered_conditional_play(self):
        """The Target exemplar's placeholder parameter must let ``GatesInBasis`` accept a
        conditional play whose argument slot holds a Boolean ``Expr``."""
        backend, _, _ = _backend_with_pulse()
        backend.register_conditional_play("x180")
        qc = QuantumCircuit(1, 1)
        flag = qc.add_input("flag", types.Bool())
        qc.conditional_play("x180", expr.logic_and(flag, qc.clbits[0]), 0)
        qc.barrier()
        with qc.if_test(expr.lift(qc.clbits[0])):
            qc.conditional_play("x180", flag, 0)
        qc.ensure_physical(backend.num_qubits)

        ensure_native_operations(qc, backend.target)  # must not raise

    def test_quantum_circuit_to_qua_rejects_non_native_operations(self):
        backend, _, _ = _backend_with_pulse()
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ensure_physical(backend.num_qubits)

        with pytest.raises(ValueError, match="not native to this backend's Target"):
            backend.quantum_circuit_to_qua(qc)

    def test_transpile_then_export_matches_the_documented_qasm(self):
        """Keeps the OpenQASM call shown in docs/backend.md honest."""
        backend, _, _ = _backend_with_pulse()
        backend.register_conditional_play("x180")
        creg = ClassicalRegister(1, "creg")
        qreg = QuantumRegister(1, "qreg")
        qc = QuantumCircuit(qreg, creg)
        qc.conditional_play("x180", condition=expr.equal(creg[0], expr.lift(False)), qubit=qreg[0])

        transpiled = transpile(qc, backend, initial_layout=[0])
        ensure_native_operations(transpiled, backend.target)

        qasm = backend.qasm3_exporter.dumps(transpiled)
        assert "box {\n  qm_conditional_play_x180(creg_0[0] == false) $0;\n}" in qasm
