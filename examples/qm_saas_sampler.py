"""
Example: Simulating circuits on the QM cloud simulator with QmSaasProvider.

This script shows how to obtain a backend from QmSaasProvider, build a simple
circuit, transpile it, and run it with the Sampler primitive on a QM SaaS
simulator instance.

Requires ``pip install qiskit-qm-provider[qm-saas]`` (``qm-saas>=1.2.0``).
Communication with the simulator is TLS-secured: the provider connects with
``QuantumMachinesManager(**instance.qmm_connection_params)``.
"""

from qiskit.circuit import QuantumCircuit
from qiskit_qm_provider import (
    QmSaasProvider,
    FluxTunableTransmonBackend,
    add_basic_macros,
    QMSamplerV2,
    QMSamplerOptions,
)
from qiskit import transpile

# Credentials are read from ~/qm_saas_config.json ({"email", "password", "host"})
# when omitted, or pass them explicitly: QmSaasProvider(email="...", password="...", host="...")
provider = QmSaasProvider()

# Spawns a simulator instance and connects a QuantumMachinesManager to it over TLS
backend = provider.get_backend(
    quam_state_folder_path="/path/to/quam/state",  # Replace with your QuAM state folder
    backend_cls=FluxTunableTransmonBackend,
)
# Add single qubit gate macros that may not yet be part of the standard Quam
add_basic_macros(backend)

# Build a simple circuit (e.g. H then measure)
qc = QuantumCircuit(1)
qc.h(0)
qc.measure_all()

try:
    # Transpile to backend and run with the Sampler (simulated on the SaaS instance)
    transpiled_circuit = transpile(qc, backend, initial_layout=[0])
    sampler = QMSamplerV2(backend, options=QMSamplerOptions(input_type=None))
    job = sampler.run([(transpiled_circuit,)])
    result = job.result()
    print(result[0].data.meas.get_counts())
finally:
    # Release the cloud simulator instance(s)
    provider.close_all()
