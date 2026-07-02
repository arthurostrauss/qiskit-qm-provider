"""Tests for sampler stream buffer layout and result assembly."""

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qiskit_qm_provider.job.stream_assembly import bit_array_from_measurement_stream, bit_array_from_stream


class TestSamplerStreamAssembly:
    def test_bit_array_from_flat_stream(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        shots = 4
        pub = SamplerPub(qc, shots=shots)
        raw = np.array([0, 1, 0, 1], dtype=int)

        bit_array = bit_array_from_measurement_stream(pub, raw, bit_width=1)

        assert bit_array.shape == pub.shape
        assert bit_array.num_shots == shots
        assert bit_array.get_counts() == {"0": 2, "1": 2}

    def test_bit_array_from_nested_stream(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)
        qc.measure(0, 0)
        shots = 3
        parameter_values = np.linspace(0, np.pi, 2)
        pub = SamplerPub.coerce((qc, parameter_values, shots))
        raw = np.array([[0, 1, 0], [1, 1, 0]], dtype=int)

        bit_array = bit_array_from_measurement_stream(pub, raw, bit_width=1)

        assert bit_array.shape == pub.shape
        assert bit_array.num_shots == shots
        assert bit_array.get_counts(loc=(0,)) == {"0": 2, "1": 1}
        assert bit_array.get_counts(loc=(1,)) == {"0": 1, "1": 2}

    def test_bit_array_from_raveled_stream(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)
        qc.measure(0, 0)
        shots = 3
        parameter_values = np.linspace(0, np.pi, 2)
        pub = SamplerPub.coerce((qc, parameter_values, shots))
        raw = np.array([0, 1, 0, 1, 1, 0], dtype=int)

        bit_array = bit_array_from_measurement_stream(pub, raw, bit_width=1)

        assert bit_array.shape == pub.shape
        assert bit_array.num_shots == shots

    def test_bit_array_stream_size_mismatch_raises(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        pub = SamplerPub(qc, shots=4)
        raw = np.array([0, 1, 0], dtype=int)

        with pytest.raises(ValueError, match="cannot reshape"):
            bit_array_from_measurement_stream(pub, raw, bit_width=1)

    def test_bit_array_from_nested_shot_stream(self):
        """QMBackend.run streams can arrive as a single nested buffer per circuit."""
        shots = 4
        raw = np.array([[0, 1, 0, 1]], dtype=int)

        bit_array = bit_array_from_stream(raw, bit_width=1, target_shape=(shots,))

        assert bit_array.shape == ()
        assert bit_array.num_shots == shots
        assert bit_array.get_counts() == {"0": 2, "1": 2}
