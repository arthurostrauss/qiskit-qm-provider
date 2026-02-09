"""
Example: Adding Qiskit Pulse calibrations to a circuit and running on the backend.

This example attaches a custom pulse-level calibration to a gate using
qc.add_calibration(). When the circuit is run (or transpiled) with the backend,
the backend picks up these calibrations and translates them to QUA.

Requires Qiskit 1.x for full Qiskit Pulse support (DriveChannel, Schedule, etc.).
"""

from qiskit.circuit import QuantumCircuit
from qiskit import transpile

# Assume backend is obtained from a provider and supports Pulse (e.g. FluxTunableTransmonBackend)
# backend = provider.get_backend(...)

# Optional: add standard macros if your Quam machine does not have them yet
# from qiskit_qm_provider.backend.backend_utils import add_basic_macros
# add_basic_macros(backend)

# Build a circuit that uses a gate we will calibrate
qc = QuantumCircuit(1)
qc.x(0)  # We will attach a custom pulse schedule to "x" on qubit 0
qc.measure_all()

# Build a Qiskit Pulse Schedule for the "x" gate on qubit 0
# (In practice you would use your backend's drive channel and a real pulse.)
try:
    from qiskit.pulse import Schedule, Play, DriveChannel
    from qiskit.pulse.library import Gaussian

    duration = 64
    sigma = duration / 4
    amp = 0.5
    x_schedule = Schedule()
    x_schedule.append(
        Play(Gaussian(duration, amp, sigma), DriveChannel(0)),
        channel=DriveChannel(0),
    )
    qc.add_calibration("x", (0,), x_schedule)
except ImportError:
    # Qiskit 2.x: Pulse is deprecated; use QMInstructionProperties for custom gates (see custom_gate.py)
    raise ImportError("This example requires Qiskit 1.x with Pulse support.")

# Transpile and run; the backend will update its calibration mapping from the circuit
transpiled = transpile(qc, backend)
job = backend.run(transpiled, shots=1024)
result = job.result()
print(result)
