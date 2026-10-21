"""Condition a registered QuAM pulse directly from a Qiskit classical expression."""

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classical import expr, types

from qm import generate_qua_script
from qiskit_qm_provider import QMProvider, add_basic_macros
provider = QMProvider()
backend = provider.get_backend()
add_basic_macros(backend)
# Register once before transpilation.  The pulse label must resolve uniquely on
# every active QuAM qubit.
backend.register_conditional_play("x180", "-y90")

q = QuantumRegister(1)
c = ClassicalRegister(1)
qc = QuantumCircuit(q, c)
x_basis = qc.add_input("basis_x", types.Bool())
qc.h(q)
qc.conditional_play("-y90", x_basis, q[0])
qc.measure(q, c)
condition = expr.equal(c, 1)
qc.conditional_play("x180", condition, q[0])

transpiled = transpile(qc, backend)
compilation = backend.quantum_circuit_to_qua(transpiled)

print(backend.qasm3_exporter.dumps(transpiled))
print("QUA:")
print(generate_qua_script(compilation.qua_program))
