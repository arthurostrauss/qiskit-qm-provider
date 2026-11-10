"""PUBs with zero bindings: valid, never compiled or sent, and their results are empty.

Qiskit's reference primitives (``StatevectorSampler``, ``StatevectorEstimator``,
``BackendEstimatorV2``) return arrays of shape ``(0,)`` for them without running anything.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_qm_provider.parameter_table import InputType, ParameterPool


@pytest.fixture
def no_qm(compilation_backend, monkeypatch):
    def refuse(_):
        raise AssertionError("a job without programs must not reach the QM")

    monkeypatch.setattr(type(compilation_backend), "qm", property(refuse))
    return compilation_backend


def _circuit(backend, measure):
    qc = QuantumCircuit(1, 1 if measure else 0)
    qc.rx(Parameter("a"), 0)
    if measure:
        qc.measure(0, 0)
    return transpile(qc, backend, optimization_level=0)


@pytest.mark.parametrize("input_type", [None, InputType.INPUT_STREAM])
def test_sampler_zero_bindings_returns_empty_result(no_qm, input_type):
    from qiskit_qm_provider.job.qm_sampler_job import QMSamplerJob

    ParameterPool.reset()
    pub = SamplerPub.coerce((_circuit(no_qm, True), np.zeros((0, 1))), shots=7)
    job = QMSamplerJob(no_qm, [pub], input_type, shots=7)
    assert job.programs == []
    job.submit()
    data = job.result()[0].data
    assert data.shape == (0,) and data.c.shape == (0,) and data.c.num_shots == 7 and data.c.num_bits == 1


def test_estimator_zero_bindings_returns_empty_result(no_qm):
    from qiskit_qm_provider.job.qm_estimator_job import QMEstimatorJob
    from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorV2

    ParameterPool.reset()
    estimator = QMEstimatorV2(no_qm)
    pubs = estimator.validate_estimator_pubs(
        [EstimatorPub.coerce((_circuit(no_qm, False), SparsePauliOp("Z"), np.zeros((0, 1))), 0.1)]
    )
    job = QMEstimatorJob(no_qm, pubs, None, switch_obs_circuit=estimator._switch_obs_circuit, abelian_grouping=True)
    assert job.programs == []
    job.submit()
    assert job.result()[0].data.evs.shape == (0,)
