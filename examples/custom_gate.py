"""
Example: Adding a custom parametric gate to the backend Target.

This demonstrates how to register a new gate (e.g. a calibrated CNOT) in the
backend so that both the Qiskit transpiler and the OpenQASM3→QUA compiler
use your QUA macro. After modifying the target, call backend.update_target().
"""

from qiskit.circuit import Parameter as QiskitParameter, Gate
from qiskit_qm_provider import QMProvider, QMInstructionProperties

# 1. Set up provider and backend (use your Quam state path or another provider)
provider = QMProvider("/path/to/quam/state")
backend = provider.get_backend()

# 2. Define an opaque parametric two-qubit gate at the circuit level
theta = QiskitParameter("theta")  # Use ASCII names
cx_cal = Gate("cx_cal", num_qubits=2, params=[theta])  # No logical definition: opaque gate

# (Optional) You may instead provide a logical definition for cx_cal so that the
# transpiler can optimize it; see the Qiskit backend transpiler interface docs.

# 3. Define the corresponding QUA macro
def qua_macro(theta_val):
    # Implement the low-level calibrated pulse sequence
    qubit_pair = backend.get_qubit_pair((0, 1))
    qubit_pair.apply("cz", amplitude_scale=theta_val)

# 4. Register the new instruction in the backend Target
duration = backend.target["cx"][(0, 1)].duration  # Reuse existing CX duration as template
properties = {
    (0, 1): QMInstructionProperties(
        duration=duration,
        qua_pulse_macro=qua_macro,
    )
}
backend.target.add_instruction(cx_cal, properties=properties)

# 5. Synchronize the internal QUA compiler mapping with the modified Target
backend.update_target()

# You can now use the "cx_cal" gate in circuits and run or compile them to QUA.
