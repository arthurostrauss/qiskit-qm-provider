"""Parameterized circuits through a real machine-derived backend and qm-qasm."""

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qm import generate_qua_script, qua

from qiskit_qm_provider import ParameterTable


@pytest.mark.parametrize("vector", [False, True], ids=["scalar", "vector"])
def test_parameterized_circuit_compiles_on_machine_backend(compilation_backend, vector):
    parameters = ParameterVector("theta", 2) if vector else [Parameter("theta")]
    circuit = QuantumCircuit(1)
    for parameter in parameters:
        circuit.rx(parameter, 0)
    circuit.measure_all()
    physical = transpile(circuit, compilation_backend, optimization_level=0)
    table = ParameterTable.from_qiskit(physical)

    with qua.program() as program:
        compiled = compilation_backend.quantum_circuit_to_qua(physical, table)

    script = generate_qua_script(program)
    assert "frame_rotation" in script
    assert '"q0.xy"' in script or "'q0.xy'" in script
    assert compiled.outputs.get_parameter("meas").length == 1
