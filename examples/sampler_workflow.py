"""
Example: Running circuits with QMSamplerV2 and the IQCC provider.

This script shows how to obtain a backend from IQCCProvider, build a simple
circuit, transpile it, and run it with the Sampler primitive.
"""

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_qm_provider import IQCCProvider
from qiskit_qm_provider import QMSamplerV2, QMSamplerOptions
from qiskit import transpile

# Set your quantum computer backend name (e.g. "qolab", "arbel")
backend_name = "arbel"

# Get provider, machine, and backend
iqcc_provider = IQCCProvider()  # Uses API token from env or pass api_token="..."
machine = iqcc_provider.get_machine(backend_name)
backend = iqcc_provider.get_backend(machine)

# Optional: inspect qubit mapping
print(backend.qubit_dict)

# Build a simple circuit (e.g. X then measure)
qc = QuantumCircuit(1)
param = Parameter("param")  # Use ASCII names for parameters (see README)
qc.h(0)
qc.measure_all()

# Transpile to backend and run with the Sampler
transpiled_circuits = transpile(qc, backend, initial_layout=[0])
sampler = QMSamplerV2(backend, options=QMSamplerOptions(input_type=None))
job = sampler.run([(transpiled_circuits,)])
result = job.result()
counts = result[
    0
].data.meas.get_counts()  # meas is the name of the DataBin (classical register associated with the measurement)
print(counts)
