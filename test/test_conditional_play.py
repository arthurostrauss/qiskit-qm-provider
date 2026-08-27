"""Tests for provider-specific conditional QUA pulse plays."""

import re
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from quam.components import Qubit
from qiskit import transpile
from qiskit.circuit import Gate, Instruction, QuantumCircuit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit.transpiler import Target

from qiskit_qm_provider import ConditionalPlay
from qiskit_qm_provider.backend.qm_backend import QMBackend
from qiskit_qm_provider.conditional_play import (
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


def _backend_with_pulse(*, get_pulse_side_effect=None):
    qubit = Mock(spec=Qubit)
    qubit.name = "q0"
    qubit.T1 = qubit.T2echo = qubit.f_01 = None
    qubit.macros = {}
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
        x180_name = conditional_play_operation_name("x180")

        assert x180_name.startswith("qm_conditional_play_x180_")
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", x180_name)
        assert conditional_play_operation_name("-y90") != conditional_play_operation_name("_y90")

    def test_circuit_method_preserves_expression_and_pulse_name(self):
        qc = QuantumCircuit(1, 1)
        condition = expr.equal(qc.clbits[0], expr.lift(False))

        qc.conditional_play("x180", condition, 0)

        operation = qc.data[0].operation
        assert isinstance(operation, ConditionalPlay)
        assert isinstance(operation, Instruction)
        assert not isinstance(operation, Gate)
        assert operation.pulse_name == "x180"
        assert operation.condition_expr == condition
        assert operation.params == [condition]

    def test_boolean_expression_is_emitted_as_gate_argument(self):
        qc = QuantumCircuit(1, 1)
        condition = expr.equal(qc.clbits[0], expr.lift(False))
        qc.conditional_play("x180", condition, 0)

        qasm = _exporter_for("x180").dumps(qc)

        assert f"{conditional_play_operation_name('x180')}(c[0] == false) q[0];" in qasm
        assert "input float" not in qasm

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
        target.add_instruction(ConditionalPlay.target_operation("x180"), {(0,): None})

        transpiled = transpile(qc, target=target)
        qasm = _exporter_for("x180").dumps(transpiled)

        assert f"{conditional_play_operation_name('x180')}(c[0] == false) $0;" in qasm
        assert transpiled.data[0].operation.pulse_name == "x180"

    def test_non_boolean_or_non_expression_condition_is_rejected(self):
        qc = QuantumCircuit(1, 1)
        with pytest.raises(CircuitError, match="classical Expr"):
            ConditionalPlay("x180", False)
        with pytest.raises(CircuitError, match="Boolean type"):
            ConditionalPlay("x180", expr.lift(1))

    def test_inverse_and_quantum_control_are_rejected(self):
        qc = QuantumCircuit(1, 1)
        operation = ConditionalPlay("x180", expr.equal(qc.clbits[0], expr.lift(False)))

        with pytest.raises(CircuitError, match="inverse"):
            operation.inverse()
        with pytest.raises(CircuitError, match="quantum-controlled"):
            operation.control()

    def test_qua_macro_resolves_the_quam_pulse_and_forwards_condition(self):
        pulse = Mock()
        qubit = Mock()
        qubit.get_pulse.return_value = pulse
        condition = object()

        _conditional_play_macro(qubit, "x180")(condition)

        qubit.get_pulse.assert_called_once_with("x180")
        pulse.play.assert_called_once_with(condition=condition)


class TestConditionalPlayRegistration:
    def test_exporter_rejects_an_unregistered_conditional_play(self):
        backend, _, _ = _backend_with_pulse()
        qc = QuantumCircuit(1, 1)
        qc.conditional_play("x180", expr.equal(qc.clbits[0], expr.lift(False)), 0)

        with pytest.raises(ValueError, match="not registered"):
            backend.qasm3_exporter.dumps(qc)

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

    def test_exporter_rejects_a_qubit_outside_the_registered_target(self):
        backend, _, _ = _backend_with_pulse()
        backend.register_conditional_play("x180")
        qc = QuantumCircuit(2, 1)
        qc.conditional_play("x180", expr.equal(qc.clbits[0], expr.lift(False)), 1)

        with pytest.raises(ValueError, match="not supported on physical qubits"):
            backend.qasm3_exporter.dumps(qc)

    @pytest.mark.parametrize("error", [ValueError("Pulse x180 not found"), ValueError("Pulse x180 is not unique")])
    def test_registration_propagates_quam_pulse_lookup_errors(self, error):
        backend, _, _ = _backend_with_pulse(get_pulse_side_effect=error)

        with pytest.raises(ValueError, match=str(error)):
            backend.register_conditional_play("x180")

        assert conditional_play_operation_name("x180") not in backend.target.operation_names
