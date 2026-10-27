"""QUA generation of circuits containing ``reset`` on the in-memory dummy QuAM.

``validate_circuits`` adds a reset to every ``backend.run``/Sampler/Estimator circuit, so
the in-memory backend's reset must compile without calibration data: it uses the thermal
reset of ``add_basic_macros`` (quam-builder's default active reset reads an unset
``rus_exit_threshold`` and fails ``Qua1GenerationPass``).
"""

import pytest
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_qm_provider.backend.backend_utils import validate_circuits

from ._in_memory_quam import build_backend

qm = pytest.importorskip("qm")
from qm import Program, generate_qua_script  # noqa: E402


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAM_STATE_PATH", str(tmp_path))
    return build_backend()


def _qua_script(backend, qc: QuantumCircuit) -> str:
    return generate_qua_script(backend.quantum_circuit_to_qua(qc).qua_program)


def test_reset_is_thermal(backend):
    for qubit in backend.machine.active_qubits:
        assert qubit.macros["reset"].reset_type == "thermalize"


@pytest.mark.parametrize("qubits", [[0], [1], [0, 1]])
def test_reset_compiles(backend, qubits):
    qc = QuantumCircuit(2)
    qc.reset(qubits)

    script = _qua_script(backend, transpile(qc, backend))

    assert "while_(" not in script


def test_validated_run_circuit_with_added_reset_compiles(backend):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    [validated] = validate_circuits(transpile(qc, backend, optimization_level=1, seed_transpiler=0))
    assert any(instruction.operation.name == "reset" for instruction in validated.data)

    _qua_script(backend, validated)


def test_estimator_pub_plans_to_a_qua_program(backend):
    """One Estimator PUB, prepared by ``validate_estimator_pubs`` (which adds the reset),
    plans to a single QUA program without hardware."""
    from qiskit_qm_provider.job.primitive_plans import EstimatorPlan
    from qiskit_qm_provider.job.qua_programs import plan_estimator_programs
    from qiskit_qm_provider.parameter_table import ParameterPool
    from qiskit_qm_provider.parameter_table.input_type import InputType
    from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions, QMEstimatorV2

    options = QMEstimatorOptions(input_type=InputType.INPUT_STREAM)
    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    transpiled = transpile(qc, backend, optimization_level=1, seed_transpiler=0)
    observables = SparsePauliOp(["ZZ", "XX"]).apply_layout(transpiled.layout)
    pub = EstimatorPub.coerce((transpiled, observables), precision=0.05)

    [prepared] = QMEstimatorV2(backend, options=options).validate_estimator_pubs([pub])
    ParameterPool.reset()
    plan = EstimatorPlan.from_pub(prepared, options, plan_index=0)

    programs, layout = plan_estimator_programs(backend, [plan])

    assert layout == [[0]]
    [program] = programs
    assert isinstance(program, Program)
