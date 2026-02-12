"""
Example: Adding Qiskit Pulse calibrations to a circuit and running on the backend.

This example attaches a custom pulse-level calibration to a gate using
qc.add_calibration(). When the circuit is run (or transpiled) with the backend,
the backend picks up these calibrations and translates them to QUA.

Requires Qiskit 1.x for full Qiskit Pulse support (DriveChannel, Schedule, etc.).
"""

from qiskit.circuit import QuantumCircuit, Gate, Parameter
from qiskit import transpile
from qiskit_qm_provider import IQCCProvider
backend = IQCCProvider().get_backend("qolab")

# Assume backend is obtained from a provider and supports Pulse (e.g. FluxTunableTransmonBackend)

# Build a circuit that uses a gate we will calibrate
qc = QuantumCircuit(1)
rx_gate = Gate("rx", 1, [Parameter("theta")])
qc.append(rx_gate, [0])
qc.measure_all()

# Build a Qiskit Pulse Schedule for the "x" gate on qubit 0
# (In practice you would use your backend's drive channel and a real pulse.)
try:
    from qiskit.pulse import Schedule, Play, DriveChannel
    from qiskit.pulse.library import Gaussian

    duration = 64
    sigma = duration / 4
    amp = 0.5
    rx_schedule = Schedule()
    rx_schedule.append(
        Play(Gaussian(duration, amp, sigma), DriveChannel(0))
    )
    qc.add_calibration(rx_gate, (0,), rx_schedule)
except ImportError:
    # Qiskit 2.x: Pulse is deprecated; use QMInstructionProperties for custom gates (see custom_gate.py)
    raise ImportError("This example requires Qiskit 1.x with Pulse support.")

# Transpile and run; the backend will update its calibration mapping from the circuit
transpiled = transpile(qc, backend)
job = backend.run(transpiled, shots=1024)
result = job.result()
print(result)
