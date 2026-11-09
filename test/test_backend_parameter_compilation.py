"""Parameterized circuits through a real machine-derived backend and qm-qasm."""

import numpy as np
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


def test_table_by_ref_compiles_on_machine_backend(compilation_backend):
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.rx(theta, 0)
    circuit.measure_all()
    physical = transpile(circuit, compilation_backend, optimization_level=0)
    table = ParameterTable.from_qiskit(physical)

    with qua.program() as program:
        table.declare()
        compiled = compilation_backend.quantum_circuit_to_qua(physical, table_by_ref=table)

    script = generate_qua_script(program)
    assert "frame_rotation" in script
    assert compiled.outputs.get_parameter("meas").length == 1


def test_openqasm_constants_are_exported(compilation_backend):
    circuit = QuantumCircuit(1)
    circuit.rx(np.pi / 2, 0)
    circuit.measure_all()
    physical = transpile(circuit, compilation_backend, optimization_level=0)
    qasm, _ = compilation_backend.qasm3_exporter.export(physical)
    assert "pi" in qasm
    assert "1.570796" not in qasm

    with qua.program() as program:
        compilation_backend.quantum_circuit_to_qua(physical)

    assert "frame_rotation" in generate_qua_script(program)


def test_get_qua_script_defaults_unbound_inputs_to_zero(compilation_backend):
    from qiskit_qm_provider import get_qua_script

    theta = Parameter("theta")
    x = ParameterVector("x", 2)
    circuit = QuantumCircuit(1)
    circuit.rz(theta, 0)
    circuit.rz(x[0] + x[1], 0)
    circuit.measure_all()
    physical = transpile(circuit, compilation_backend, optimization_level=0)
    assert "frame_rotation" in get_qua_script(compilation_backend, physical)
    assert "frame_rotation" in get_qua_script(compilation_backend, physical, param_table={"theta": 0.25, "x": [0.1, 0.2]})


def test_get_qua_script_rejects_tables(compilation_backend):
    from qiskit_qm_provider import get_qua_script

    circuit = QuantumCircuit(1)
    circuit.rz(Parameter("theta"), 0)
    physical = transpile(circuit, compilation_backend, optimization_level=0)
    with pytest.raises(TypeError, match="dict of Python values"):
        get_qua_script(compilation_backend, physical, param_table=ParameterTable.from_qiskit(physical))
