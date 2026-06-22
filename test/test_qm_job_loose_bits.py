"""Tests for QMJob result assembly with loose classical bits."""

from qiskit.circuit import ClassicalRegister, Clbit, QuantumCircuit

from qiskit_qm_provider.backend.backend_utils import measurement_output_bit_sizes


class TestQMJobLooseBitStreamKeys:
    def test_stream_keys_match_qua_program_naming(self):
        """Result assembly must fetch the same stream names that get_run_program saves."""
        creg = ClassicalRegister(1, "c")
        loose = Clbit()
        qc = QuantumCircuit(2)
        qc.add_register(creg)
        qc.add_bits([loose])
        qc.measure(0, creg[0])
        qc.measure(1, loose)

        circuit_index = 0
        stream_keys = [f"{name}_{circuit_index}" for name in measurement_output_bit_sizes(qc)]

        assert stream_keys == ["c_0", "_bit0_0"]
