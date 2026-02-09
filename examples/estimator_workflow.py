"""
Example: Running expectation-value jobs with QMEstimatorV2.

This script shows how to use the Estimator primitive with optional real-time
parameter input (e.g. Input Streaming or DGX Quantum).
"""

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_qm_provider import QMEstimatorV2, QMEstimatorOptions, InputType
from qiskit import transpile
import numpy as np
# Get a backend (example: IQCC; or use QMProvider, QmSaasProvider)
from qiskit_qm_provider import IQCCProvider
backend = IQCCProvider().get_backend("qolab")  # Replace with your backend name

# Build a parametric circuit and observables
theta = Parameter("theta")  # Use ASCII names
circuit = QuantumCircuit(1)
circuit.rx(theta, 0)

observables = SparsePauliOp(["Z"])
parameter_values =  np.linspace(0, np.pi, 10) # Values for theta per run

# Transpile to backend
circuit = transpile(circuit, backend)
observables = observables.apply_layout(circuit.layout)
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

evs = result[0].data.evs
stds = result[0].data.stds
print("Expectation values: ", evs)
print("Standard errors: ", stds)
