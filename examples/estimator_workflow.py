"""
Example: Running expectation-value jobs with QMEstimatorV2.

This script shows how to use the Estimator primitive with optional real-time
parameter input (e.g. Input Streaming or DGX Quantum).
"""

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_qm_provider import QMEstimatorV2, QMEstimatorOptions, InputType
from qiskit import transpile

# Get a backend (example: IQCC; or use QMProvider, QmSaasProvider)
from qiskit_qm_provider import IQCCProvider
backend = IQCCProvider().get_backend("qolab")  # Replace with your backend name

# Build a parametric circuit and observables
theta = Parameter("theta")  # Use ASCII names
circuit = QuantumCircuit(1)
circuit.h(0)
circuit.rz(theta, 0)
circuit.h(0)

observables = SparsePauliOp(["Z"])
parameter_values = [[0.0], [1.0], [2.0]]  # Values for theta per run

# Transpile to backend
circuit = transpile(circuit, backend)

# Option 1: Preload parameters at compile time (input_type=None)
options = QMEstimatorOptions(input_type=None)
estimator = QMEstimatorV2(backend=backend, options=options)
job = estimator.run([(circuit, observables, parameter_values)])
result = job.result()

# Option 2: Real-time parameter updates via Input Streaming
# options = QMEstimatorOptions(input_type=InputType.INPUT_STREAM)
# estimator = QMEstimatorV2(backend=backend, options=options)
# job = estimator.run([(circuit, observables, parameter_values)])
# result = job.result()

print(result)
