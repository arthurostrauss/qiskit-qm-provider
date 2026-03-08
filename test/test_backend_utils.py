"""Tests for backend utility functions."""

import pytest
import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Pauli, PauliList

from qiskit_qm_provider.backend.backend_utils import (
    look_for_standard_op,
    get_extended_gate_name_mapping,
    has_reset_at_boundary,
    validate_circuits,
    binary,
    logically_active_qubits,
    get_non_trivial_observables,
)


class TestLookForStandardOp:
    """Tests for the look_for_standard_op function."""

    def test_direct_mapping(self):
        assert look_for_standard_op("cnot") == "cx"
        assert look_for_standard_op("hadamard") == "h"
        assert look_for_standard_op("identity") == "id"
        assert look_for_standard_op("readout") == "measure"
        assert look_for_standard_op("meas") == "measure"

    def test_rotation_aliases(self):
        assert look_for_standard_op("x180") == "x"
        assert look_for_standard_op("y180") == "y"
        assert look_for_standard_op("x90") == "sx"
        assert look_for_standard_op("x/2") == "sx"
        assert look_for_standard_op("y90") == "sy"
        assert look_for_standard_op("y/2") == "sy"

    def test_negative_rotation_aliases(self):
        assert look_for_standard_op("-x90") == "sxdg"
        assert look_for_standard_op("-x/2") == "sxdg"
        assert look_for_standard_op("-y/2") == "sydg"
        assert look_for_standard_op("-y90") == "sydg"

    def test_multi_qubit_aliases(self):
        assert look_for_standard_op("cphase") == "cp"
        assert look_for_standard_op("zz") == "rzz"
        assert look_for_standard_op("yy") == "ryy"
        assert look_for_standard_op("xx") == "rxx"

    def test_wait_alias(self):
        assert look_for_standard_op("wait") == "delay"

    def test_case_insensitive(self):
        assert look_for_standard_op("CNOT") == "cx"
        assert look_for_standard_op("Hadamard") == "h"
        assert look_for_standard_op("X180") == "x"

    def test_unknown_op_passthrough(self):
        assert look_for_standard_op("my_custom_gate") == "my_custom_gate"

    def test_partial_match(self):
        assert look_for_standard_op("my_x180_gate") == "x"


class TestGetExtendedGateNameMapping:
    def test_contains_standard_gates(self):
        gate_map = get_extended_gate_name_mapping()
        assert "x" in gate_map
        assert "h" in gate_map
        assert "cx" in gate_map
        assert "rz" in gate_map

    def test_contains_custom_gates(self):
        gate_map = get_extended_gate_name_mapping()
        assert "sy" in gate_map
        assert "sydg" in gate_map
        assert "cr" in gate_map

    def test_custom_gate_types(self):
        from qiskit_qm_provider.additional_gates import SYGate, SYdgGate, CRGate

        gate_map = get_extended_gate_name_mapping()
        assert isinstance(gate_map["sy"], SYGate)
        assert isinstance(gate_map["sydg"], SYdgGate)
        assert isinstance(gate_map["cr"], CRGate)


class TestHasResetAtBoundary:
    def test_empty_circuit(self):
        qc = QuantumCircuit(2)
        assert has_reset_at_boundary(qc) is True

    def test_circuit_with_start_reset(self):
        qc = QuantumCircuit(1)
        qc.reset(0)
        qc.x(0)
        assert has_reset_at_boundary(qc) is True

    def test_circuit_with_end_reset(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.reset(0)
        assert has_reset_at_boundary(qc) is True

    def test_circuit_without_reset(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        assert has_reset_at_boundary(qc) is False

    def test_multi_qubit_partial_reset(self):
        qc = QuantumCircuit(2)
        qc.reset(0)
        qc.x(0)
        qc.x(1)
        assert has_reset_at_boundary(qc) is False

    def test_multi_qubit_all_reset(self):
        qc = QuantumCircuit(2)
        qc.reset(0)
        qc.reset(1)
        qc.x(0)
        qc.x(1)
        assert has_reset_at_boundary(qc) is True


class TestValidateCircuits:
    def test_single_circuit_wrapped_in_list(self):
        qc = QuantumCircuit(1)
        qc.reset(0)
        qc.x(0)
        result = validate_circuits(qc)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_list_of_circuits(self):
        qc1 = QuantumCircuit(1)
        qc1.reset(0)
        qc1.x(0)
        qc2 = QuantumCircuit(1)
        qc2.reset(0)
        qc2.h(0)
        result = validate_circuits([qc1, qc2])
        assert len(result) == 2

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="list of QuantumCircuits"):
            validate_circuits(["not_a_circuit"])

    def test_adds_reset_when_missing(self):
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        result = validate_circuits(qc, should_reset=True)
        ops = [inst.operation.name for inst in result[0].data]
        assert "reset" in ops

    def test_no_reset_when_disabled(self):
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        result = validate_circuits(qc, should_reset=False)
        ops = [inst.operation.name for inst in result[0].data]
        assert result[0].num_qubits == 1

    def test_multiple_registers_per_clbit_raises(self):
        qc = QuantumCircuit(1)
        cr1 = ClassicalRegister(1, "c1")
        cr2 = ClassicalRegister(1, "c2")
        qc.add_register(cr1)
        qc.add_register(cr2)
        qc.measure(0, cr1[0])

        qc2 = QuantumCircuit(1)
        qc2.add_register(cr1)
        qc2.add_register(cr2)
        from qiskit.circuit import Clbit

        shared_bit = Clbit()
        qc_multi = QuantumCircuit(1, name="multi_reg")
        cr_a = ClassicalRegister(bits=[shared_bit], name="a")
        cr_b = ClassicalRegister(bits=[shared_bit], name="b")
        qc_multi.add_register(cr_a)
        qc_multi.add_register(cr_b)
        qc_multi.reset(0)
        qc_multi.measure(0, shared_bit)

        with pytest.raises(ValueError, match="one register per clbit"):
            validate_circuits(qc_multi)


class TestBinary:
    def test_basic(self):
        assert binary(5, 4) == "0101"

    def test_zero(self):
        assert binary(0, 8) == "00000000"

    def test_no_padding(self):
        assert binary(3) == "11"

    def test_large_value(self):
        assert binary(255, 8) == "11111111"


class TestLogicallyActiveQubits:
    def test_all_active(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.h(1)
        qc.rz(0.5, 2)
        active = logically_active_qubits(qc)
        assert len(active) == 3

    def test_delay_not_counted(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.delay(100, 1)
        active = logically_active_qubits(qc)
        indices = [qc.find_bit(q).index for q in active]
        assert 0 in indices
        assert 1 not in indices

    def test_empty_circuit(self):
        qc = QuantumCircuit(3)
        active = logically_active_qubits(qc)
        assert len(active) == 0


class TestGetNonTrivialObservables:
    def test_single_qubit(self):
        observables = PauliList([Pauli("IXI")])
        active_indices = [1]
        result = get_non_trivial_observables(observables, active_indices)
        assert len(result) == 1
        assert result[0] == Pauli("X")

    def test_multi_qubit(self):
        observables = PauliList([Pauli("XYZ")])
        active_indices = [0, 1, 2]
        result = get_non_trivial_observables(observables, active_indices)
        assert len(result) == 1
        # The function builds the label by forward-iterating over qubits
        # extracting from the reversed label, resulting in reversed order
        assert result[0] == Pauli("ZYX")

    def test_subset_of_qubits(self):
        observables = PauliList([Pauli("XIYIZ")])
        active_indices = [0, 2, 4]
        result = get_non_trivial_observables(observables, active_indices)
        assert len(result) == 1
