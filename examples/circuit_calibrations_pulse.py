"""
Example: Adding Qiskit Pulse calibrations to a circuit and running on the backend.

This example attaches a custom pulse-level calibration to a gate using
qc.add_calibration(). When the circuit is run (or transpiled) with the backend,
the backend picks up these calibrations and translates them to QUA.

Requires Qiskit 1.x for full Qiskit Pulse support (DriveChannel, Schedule, etc.).
"""

# %%
from qiskit.circuit import QuantumCircuit, Parameter, duration
from qiskit import transpile
from qiskit_qm_provider import IQCCProvider
import numpy as np

try:
    import qiskit.pulse as qp
except ImportError as e:
    # Qiskit 2.x: Pulse is deprecated; use QMInstructionProperties for custom gates (see custom_gate.py)
    raise ImportError("This example requires Qiskit 1.x with Pulse support.") from e

backend = IQCCProvider().get_backend("arbel")

# Assume backend is obtained from a provider and supports Pulse (e.g. FluxTunableTransmonBackend)
physical_qubit = (0,)  # Specify qubit in Qiskit through indices.
qubit = backend.get_qubit(physical_qubit[0])

# Build a circuit that uses a gate we will calibrate
p = Parameter("theta")
qc = QuantumCircuit(1)
qc.rx(p, 0)
qc.measure_all()

# Duration can be fetched from target directly for PulseMacros
ref_duration_sec = backend.target["x"][(0,)].duration
ref_duration_dt = duration.duration_in_dt(ref_duration_sec, backend.dt)

# Reference amplitude: needs to be fetched from Quam directly
ref_amp = qubit.get_pulse("x180").amplitude
ref_beta = qubit.get_pulse("x180").alpha

# Build a Qiskit Pulse Schedule for the "x" gate on qubit 0
# (In practice you would use your backend's drive channel and a real pulse.)
with qp.build(backend=backend, name="rx_cal") as rx_schedule:
    qp.play(
        qp.Drag(
            duration=ref_duration_dt, amp=ref_amp / np.pi * p, sigma=40, beta=ref_beta
        ),
        backend.drive_channel(physical_qubit[0]),
    )

qc.add_calibration("rx", (0,), rx_schedule, params=[p])
transpiled = transpile(qc, backend)
# %%
# Transpile and run; the backend will update its calibration mapping from the circuit

job = backend.run(transpiled.assign_parameters({p: np.pi}), shots=1024)
result = job.result()
print(result)

# %%
