"""Tests for estimator stream buffer layout and result assembly."""

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_qm_provider.job.qm_estimator_job import (
    _ExecutionPlan,
    counts_from_estimator_stream,
)
from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions


class TestEstimatorStreamAssembly:
    def test_stream_buffer_count_with_compile_time_parameters(self):
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 10)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.015625)
        plan = _ExecutionPlan.from_pub(pub, QMEstimatorOptions(input_type=None))

        assert plan.total_tasks == 10
        assert plan.stream_buffer_count == 10
        assert plan.stream_indices_for_metadata() == list(range(10))

    def test_stream_buffer_count_with_streamed_parameters(self):
        from qiskit_qm_provider.parameter_table import InputType

        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 10)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.015625)
        plan = _ExecutionPlan.from_pub(pub, QMEstimatorOptions(input_type=InputType.INPUT_STREAM))

        assert plan.total_tasks == 10
        assert plan.stream_buffer_count == 10
        assert plan.stream_indices_for_metadata() == list(range(10))

    def test_counts_from_estimator_stream_selects_matching_rows(self):
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 3)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.5)
        plan = _ExecutionPlan.from_pub(pub, QMEstimatorOptions(input_type=None))

        shots = plan.shots
        stream_count = plan.stream_buffer_count
        raw = np.zeros(stream_count * shots, dtype=int)
        for stream_idx in plan.stream_indices_for_metadata():
            raw[stream_idx * shots] = 1

        counts_list = counts_from_estimator_stream(plan, raw)
        assert len(counts_list) == len(plan.metadata)
        for counts in counts_list:
            assert counts["0"] + counts.get("1", 0) == shots

    def test_counts_from_estimator_stream_nested_buffers(self):
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 3)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.5)
        plan = _ExecutionPlan.from_pub(pub, QMEstimatorOptions(input_type=None))

        shots = plan.shots
        stream_count = plan.stream_buffer_count
        raw = np.zeros((stream_count, shots), dtype=int)
        for stream_idx in plan.stream_indices_for_metadata():
            raw[stream_idx, 0] = 1

        counts_list = counts_from_estimator_stream(plan, raw)
        assert len(counts_list) == len(plan.metadata)

    def test_counts_from_estimator_stream_streamed_parameters(self):
        from qiskit_qm_provider.parameter_table import InputType

        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)
        observables = SparsePauliOp(["Z"])
        parameter_values = np.linspace(0, np.pi, 3)
        pub = EstimatorPub.coerce((circuit, observables, parameter_values), precision=0.5)
        plan = _ExecutionPlan.from_pub(pub, QMEstimatorOptions(input_type=InputType.INPUT_STREAM))

        shots = plan.shots
        raw = np.zeros(plan.stream_buffer_count * shots, dtype=int)
        for stream_idx in plan.stream_indices_for_metadata():
            raw[stream_idx * shots] = 1

        counts_list = counts_from_estimator_stream(plan, raw)
        assert len(counts_list) == len(plan.metadata)
