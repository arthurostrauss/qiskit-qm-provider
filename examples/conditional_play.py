"""Condition a registered QuAM pulse directly from a Qiskit classical expression."""

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit_qm_provider import QMProvider


provider = QMProvider("/path/to/quam/state")
backend = provider.get_backend()

# Register once before transpilation.  The pulse label must resolve uniquely on
# every active QuAM qubit.
backend.register_conditional_play("x180")

qc = QuantumCircuit(1, 1)
condition = expr.equal(qc.clbits[0], expr.lift(False))
qc.conditional_play("x180", condition, 0)

transpiled = transpile(qc, backend)
compilation = backend.quantum_circuit_to_qua(transpiled)
