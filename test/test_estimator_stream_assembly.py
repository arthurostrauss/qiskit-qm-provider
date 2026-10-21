"""Tests for estimator stream buffer layout and result assembly."""

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit
from qiskit.circuit.classical import types
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_qm_provider.job.primitive_plans import EstimatorPlan
from qiskit_qm_provider.job.qm_estimator_job import counts_from_estimator_stream
from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions


def _rx_circuit():
    """One-qubit ``rx(theta)`` circuit with the estimator readout (``__c`` and ``obs_0``)."""
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("theta"), 0)
    circuit.add_register(ClassicalRegister(1, "__c"))
    circuit.add_input("obs_0", types.Uint(4))
    circuit.measure(0, 0)
    return circuit


class TestEstimatorStreamAssembly:
    def test_one_buffer_per_binding_with_compile_time_parameters(self):
        circuit = _rx_circuit()
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 10)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.015625)
        plan = EstimatorPlan.from_pub(pub, QMEstimatorOptions(input_type=None))

        assert len(plan.metadata) == 10

    def test_one_buffer_per_binding_with_streamed_parameters(self):
        from qiskit_qm_provider.parameter_table import InputType

        circuit = _rx_circuit()
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 10)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.015625)
        plan = EstimatorPlan.from_pub(pub, QMEstimatorOptions(input_type=InputType.INPUT_STREAM))

        assert len(plan.metadata) == 10

    def test_counts_from_estimator_stream_selects_matching_rows(self):
        circuit = _rx_circuit()
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 3)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.5)
        plan = EstimatorPlan.from_pub(pub, QMEstimatorOptions(input_type=None))

        shots = plan.shots
        stream_count = len(plan.metadata)
        raw = np.zeros(stream_count * shots, dtype=int)
        for stream_idx in range(len(plan.metadata)):
            raw[stream_idx * shots] = 1

        counts_list = counts_from_estimator_stream(plan, raw)
        assert len(counts_list) == len(plan.metadata)
        for counts in counts_list:
            assert counts["0"] + counts.get("1", 0) == shots

    def test_counts_from_estimator_stream_nested_buffers(self):
        circuit = _rx_circuit()
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 3)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.5)
        plan = EstimatorPlan.from_pub(pub, QMEstimatorOptions(input_type=None))

        shots = plan.shots
        stream_count = len(plan.metadata)
        raw = np.zeros((stream_count, shots), dtype=int)
        for stream_idx in range(len(plan.metadata)):
            raw[stream_idx, 0] = 1

        counts_list = counts_from_estimator_stream(plan, raw)
        assert len(counts_list) == len(plan.metadata)

    def test_counts_from_estimator_stream_streamed_parameters(self):
        from qiskit_qm_provider.parameter_table import InputType

        circuit = _rx_circuit()
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 3)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.5)
        plan = EstimatorPlan.from_pub(pub, QMEstimatorOptions(input_type=InputType.INPUT_STREAM))

        shots = plan.shots
        raw = np.zeros(len(plan.metadata) * shots, dtype=int)
        for stream_idx in range(len(plan.metadata)):
            raw[stream_idx * shots] = 1

        counts_list = counts_from_estimator_stream(plan, raw)
        assert len(counts_list) == len(plan.metadata)
