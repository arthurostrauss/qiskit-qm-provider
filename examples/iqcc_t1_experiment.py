"""
Example: Running a Qiskit Experiments T1 characterization with IQCCProvider.

This script fetches a backend from the Israeli Quantum Computing Center (IQCC)
and runs a T1 relaxation-time experiment from the qiskit-experiments library.
Requires: qiskit-experiments, iqcc_cloud_client, and IQCC API access.
"""

from qiskit_qm_provider import IQCCProvider

# Backend name at IQCC (e.g. "qolab", "arbel")
backend_name = "qolab"

# Get backend from IQCC (option 1: by name; option 2: get_machine then get_backend)
iqcc_provider = IQCCProvider()  # api_token=... or set IQCC_API_TOKEN
backend = iqcc_provider.get_backend(backend_name)

# Run T1 experiment using Qiskit Experiments
try:
    from qiskit_experiments.library import T1
except ImportError as e:
    raise ImportError("This example requires qiskit-experiments: pip install qiskit-experiments") from e

# Physical qubit(s) to characterize and delay range (seconds)
physical_qubits = (0,)
delays = [10e-6, 20e-6, 50e-6, 100e-6]  # Example range; adjust to your backend (min 3 points)

t1_exp = T1(physical_qubits=physical_qubits, backend=backend, delays=delays)

# Run and analyze
t1_data = t1_exp.run(backend_run=True)
print(t1_data)
result = t1_data.block_for_results()

print(f"T1 result: {result}")
