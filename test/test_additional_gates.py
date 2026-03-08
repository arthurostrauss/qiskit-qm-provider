"""Tests for additional gates (SYGate, SYdgGate, CRGate)."""

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate, RZXGate

from qiskit_qm_provider.additional_gates import SYGate, SYdgGate, CRGate


class TestSYGate:
    def test_name(self):
        gate = SYGate()
        assert gate.name == "sy"

    def test_num_qubits(self):
        gate = SYGate()
        assert gate.num_qubits == 1

    def test_num_params(self):
        gate = SYGate()
        assert len(gate.params) == 0

    def test_definition(self):
        gate = SYGate()
        gate._define()
        defn = gate.definition
        assert defn is not None
        assert defn.num_qubits == 1
        ops = [inst.operation.name for inst in defn.data]
        assert "ry" in ops

    def test_inverse(self):
        gate = SYGate()
        inv = gate.inverse()
        assert inv is not None

    def test_power(self):
        """power() delegates to gate_map()['ry'] which returns an instance;
        calling an instance with an angle raises TypeError -- track as known issue."""
        gate = SYGate()
        with pytest.raises(TypeError):
            gate.power(2)

    def test_equality(self):
        g1 = SYGate()
        g2 = SYGate()
        assert g1 == g2

    def test_inequality(self):
        g1 = SYGate()
        g2 = SYdgGate()
        assert g1 != g2

    def test_array(self):
        gate = SYGate()
        arr = np.array(gate)
        assert arr.shape == (2, 2)
        expected = np.array(RYGate(np.pi / 2))
        np.testing.assert_array_almost_equal(arr, expected)

    def test_label(self):
        gate = SYGate(label="my_sy")
        assert gate.label == "my_sy"


class TestSYdgGate:
    def test_name(self):
        gate = SYdgGate()
        assert gate.name == "sydg"

    def test_num_qubits(self):
        gate = SYdgGate()
        assert gate.num_qubits == 1

    def test_definition(self):
        gate = SYdgGate()
        gate._define()
        defn = gate.definition
        assert defn is not None
        ops = [inst.operation.name for inst in defn.data]
        assert "ry" in ops

    def test_inverse(self):
        gate = SYdgGate()
        inv = gate.inverse()
        assert inv is not None

    def test_power(self):
        """power() delegates to gate_map()['ry'] which returns an instance;
        calling an instance with an angle raises TypeError -- track as known issue."""
        gate = SYdgGate()
        with pytest.raises(TypeError):
            gate.power(2)

    def test_equality(self):
        g1 = SYdgGate()
        g2 = SYdgGate()
        assert g1 == g2

    def test_array(self):
        gate = SYdgGate()
        arr = np.array(gate)
        assert arr.shape == (2, 2)
        expected = np.array(RYGate(-np.pi / 2))
        np.testing.assert_array_almost_equal(arr, expected)

    def test_sy_times_sydg_is_identity(self):
        sy = np.array(SYGate())
        sydg = np.array(SYdgGate())
        product = sy @ sydg
        np.testing.assert_array_almost_equal(product, np.eye(2))


class TestCRGate:
    def test_name(self):
        gate = CRGate()
        assert gate.name == "cr"

    def test_num_qubits(self):
        gate = CRGate()
        assert gate.num_qubits == 2

    def test_definition(self):
        gate = CRGate()
        gate._define()
        defn = gate.definition
        assert defn is not None
        assert defn.num_qubits == 2

    def test_inverse(self):
        gate = CRGate()
        inv = gate.inverse()
        assert isinstance(inv, RZXGate)

    def test_power(self):
        """power() delegates to gate_map()['rzx'] which returns an instance;
        calling an instance with an angle raises TypeError -- track as known issue."""
        gate = CRGate()
        with pytest.raises(TypeError):
            gate.power(2)

    def test_equality(self):
        g1 = CRGate()
        g2 = CRGate()
        assert g1 == g2

    def test_array(self):
        gate = CRGate()
        arr = np.array(gate)
        assert arr.shape == (4, 4)
        expected = np.array(RZXGate(np.pi / 2))
        np.testing.assert_array_almost_equal(arr, expected)


class TestMonkeyPatching:
    def test_sy_method_on_circuit(self):
        qc = QuantumCircuit(1)
        qc.sy(0)
        assert len(qc.data) == 1
        assert qc.data[0].operation.name == "sy"

    def test_sydg_method_on_circuit(self):
        qc = QuantumCircuit(1)
        qc.sydg(0)
        assert len(qc.data) == 1
        assert qc.data[0].operation.name == "sydg"

    def test_cr_method_on_circuit(self):
        qc = QuantumCircuit(2)
        qc.cr(0, 1)
        assert len(qc.data) == 1
        assert qc.data[0].operation.name == "cr"
