"""Tests for Qiskit array support (``arthurostrauss/qiskit@full_real_time_expr``).

Both Qiskit variants are covered: table logic runs in both modes by monkeypatching
:func:`qiskit_has_array`; tests that need the fork's exporter / ``types.Array`` are skipped on
stock Qiskit, and tests that need the stock per-element exporter are skipped on the fork.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter as QiskitParameter, ParameterVector
from qiskit.circuit.classical import expr, types
from qiskit.qasm3 import Exporter
from qm import qua, generate_qua_script

from qiskit_qm_provider import InputType, ParameterPool, ParameterTable, QUA2DArray
from qiskit_qm_provider.backend.qm_backend import QMBackend
from qiskit_qm_provider.parameter_table import qiskit_classical
from qiskit_qm_provider.parameter_table.qiskit_classical import qiskit_has_array

HAS_ARRAY = hasattr(types, "Array")
fork_only = pytest.mark.skipif(not HAS_ARRAY, reason="requires Qiskit with types.Array (full_real_time_expr)")
stock_only = pytest.mark.skipif(HAS_ARRAY, reason="requires stock Qiskit exporter (per-element inputs)")
BASIS = ["rx", "rz"]


@pytest.fixture(autouse=True)
def reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


@pytest.fixture(params=[True, False], ids=["array_mode", "per_element_mode"])
def array_mode(request, monkeypatch):
    monkeypatch.setattr(qiskit_classical, "qiskit_has_array", lambda: request.param)
    return request.param


def _exporter():
    return Exporter(includes=(), basis_gates=BASIS, disable_constants=True)


def _partial_vector_circuit(name="t", size=5):
    qc = QuantumCircuit(1)
    vec = ParameterVector(name, size)
    qc.rx(vec[2], 0)
    return qc, vec


class _RecordingCompiler:
    """Stands in for ``qm_qasm.Compiler``; records what the provider sends."""

    def __init__(self):
        self.calls = []

    def compile(self, qasm, compilation_name=None, inputs=None):
        self.calls.append(SimpleNamespace(qasm=qasm, name=compilation_name, inputs=inputs))
        return SimpleNamespace(name=compilation_name, result_program=None, uuid="0", source_code=qasm)


def _real_compiler():
    from qm_qasm import Compiler, HardwareConfig, OperationIdentifier

    ops = {
        OperationIdentifier("rx", 1, (0,)): lambda t: qua.frame_rotation(t, "q0"),
        OperationIdentifier("rz", 1, (0,)): lambda t: qua.frame_rotation(t, "q0"),
    }
    return Compiler(hardware_config=HardwareConfig(quantum_operations_db=ops, physical_qubits={0: ("q0",)}))


def _compile(qc, table, compiler):
    fake_backend = SimpleNamespace(qm_qasm_basis_gates=BASIS, compiler=compiler)
    return QMBackend.quantum_circuit_to_qua(fake_backend, qc, table)


# ---------------------------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------------------------


def test_detection_matches_types_array():
    assert qiskit_has_array() is HAS_ARRAY


def test_qiskit_supports_array_keyword_removed():
    qc, _ = _partial_vector_circuit()
    with pytest.raises(TypeError):
        ParameterTable.from_qiskit(qc, qiskit_supports_array=False)


# ---------------------------------------------------------------------------------------------
# ParameterTable.from_qiskit -- both variants
# ---------------------------------------------------------------------------------------------


def test_partially_used_vector_layout(array_mode):
    qc, vec = _partial_vector_circuit()
    table = ParameterTable.from_qiskit(qc)
    if array_mode:
        assert list(table.table) == ["t"]
        assert table.get_parameter("t").length == 5
    else:
        assert list(table.table) == [f"_t_{i}_" for i in range(5)]


def test_filter_split_vector(array_mode):
    qc = QuantumCircuit(1)
    vec = ParameterVector("t", 3)
    for p in vec:
        qc.rx(p, 0)
    keep_first = lambda p: getattr(p, "index", None) == 0  # noqa: E731
    table = ParameterTable.from_qiskit(qc, filter_function=keep_first)
    # array mode: the full vector is declared; stock: all elements (as on main)
    assert list(table.table) == (["t"] if array_mode else [f"_t_{i}_" for i in range(3)])


def test_filter_whole_vector(array_mode):
    qc = QuantumCircuit(1)
    vec = ParameterVector("t", 2)
    other = QiskitParameter("a")
    qc.rx(vec[0], 0)
    qc.rx(vec[1], 0)
    qc.rz(other, 0)
    table = ParameterTable.from_qiskit(qc, filter_function=lambda p: p.name != "a", name="keep_t")
    assert list(table.table) == (["t"] if array_mode else ["_t_0_", "_t_1_"])
    table = ParameterTable.from_qiskit(qc, filter_function=lambda p: p.name == "a", name="keep_a")
    assert list(table.table) == ["a"]


@pytest.mark.parametrize("make_type", ["duration", "stretch"])
def test_duration_stretch_input_rejected(make_type):
    ty = types.Duration() if make_type == "duration" else getattr(types, "Stretch", lambda: None)()
    if ty is None:
        pytest.skip("types.Stretch unavailable")
    qc = QuantumCircuit(1)
    try:
        qc.add_input("d", ty)
    except Exception as exc:  # Qiskit itself may refuse this input type
        pytest.skip(f"Qiskit refuses {make_type} input: {exc}")
    with pytest.raises(ValueError, match="Unsupported classical type"):
        ParameterTable.from_qiskit(qc)
    with pytest.raises(ValueError, match="Unsupported classical type"):
        _compile(qc, None, _RecordingCompiler())


def test_duration_declared_var_rejected_in_compile():
    qc = QuantumCircuit(1)
    qc.add_var("d", expr.Value(10, types.Duration()))
    with pytest.raises(ValueError, match="Unsupported classical type"):
        _compile(qc, None, _RecordingCompiler())


# ---------------------------------------------------------------------------------------------
# Fork-only: types.Array inputs
# ---------------------------------------------------------------------------------------------


@fork_only
def test_array_input_var_maps_to_array_parameter():
    qc = QuantumCircuit(1)
    qc.add_input("arr", types.Array(types.Uint(8), 3))
    table = ParameterTable.from_qiskit(qc)
    param = table.get_parameter("arr")
    assert param.length == 3 and param.type is int


@fork_only
def test_zero_size_array_rejected():
    qc = QuantumCircuit(1)
    qc.add_input("arr", types.Array(types.Float(), 0))
    with pytest.raises(ValueError, match="zero-size"):
        ParameterTable.from_qiskit(qc)
    with pytest.raises(ValueError, match="zero-size"):
        _compile(qc, None, _RecordingCompiler())


@fork_only
def test_duration_array_rejected():
    qc = QuantumCircuit(1)
    qc.add_input("arr", types.Array(types.Duration(), 2))
    with pytest.raises(ValueError, match="Unsupported classical type"):
        ParameterTable.from_qiskit(qc)


# ---------------------------------------------------------------------------------------------
# End-to-end through quantum_circuit_to_qua
# ---------------------------------------------------------------------------------------------


@fork_only
def test_e2e_fork_partial_vector_with_qua2d_row():
    qc, vec = _partial_vector_circuit(name="x")
    table = ParameterTable.from_qiskit(qc)
    grid = QUA2DArray("grid", [[0.1] * 5, [0.2] * 5])
    compiler = _RecordingCompiler()
    with qua.program() as prog:
        table.declare(pause_program=False)
        grid.declare()
        table.get_parameter("x").assign(grid[1])
        _compile(qc, table, compiler)
        assert compiler.calls[0].inputs["x"] is table.get_parameter("x").var
    call = compiler.calls[0]
    assert "input array[float[64], 5] x;" in call.qasm
    assert list(call.inputs) == ["x"]
    assert "for_(" in generate_qua_script(prog)


@fork_only
def test_e2e_fork_real_compiler():
    from packaging.version import Version
    from importlib.metadata import version

    if Version(version("qm-qasm")) < Version("1.8.0"):
        pytest.skip("array inputs need qm-qasm >= 1.8.0")
    qc, _ = _partial_vector_circuit()
    table = ParameterTable.from_qiskit(qc)
    with qua.program():
        _compile(qc, table, _real_compiler())


@stock_only
def test_e2e_stock_real_compiler():
    qc, _ = _partial_vector_circuit()
    qc.rz(QiskitParameter("a"), 0)
    table = ParameterTable.from_qiskit(qc)
    with qua.program() as prog:
        _compile(qc, table, _real_compiler())
    assert "frame_rotation" in generate_qua_script(prog)


def test_qua2d_row_assign_length_mismatch():
    from qiskit_qm_provider import Parameter

    p = Parameter("p", [0.0] * 3)
    grid = QUA2DArray("grid2", [[0.1] * 4, [0.2] * 4])
    with qua.program():
        p.declare()
        grid.declare()
        with pytest.raises(ValueError, match="Mismatch in length"):
            p.assign(grid[0])


def _mixed_vector_pub():
    from qiskit.primitives.containers.bindings_array import BindingsArray
    from qiskit.primitives.containers.sampler_pub import SamplerPub

    qc, vec = _partial_vector_circuit(name="theta", size=4)
    scalar = QiskitParameter("a")
    qc.rz(scalar, 0)
    # Deliberately reverse circuit parameter order in the PUB.
    return SamplerPub(qc, BindingsArray({(vec[2], scalar): [[0.2, 0.1]]}), shots=2)


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
def test_vector_pub_local_streaming(array_mode, primitive, monkeypatch):
    from qiskit_qm_provider.job.qm_sampler_job import QMSamplerJob
    from qiskit_qm_provider.job.qm_estimator_job import QMEstimatorJob

    pub = _mixed_vector_pub()
    table = ParameterTable.from_qiskit(pub.circuit, input_type=InputType.INPUT_STREAM)
    pushed = []
    monkeypatch.setattr(table, "push_to_opx", lambda values, *args: pushed.append(values))
    backend = SimpleNamespace(qm=None)
    if primitive == "sampler":
        job = SimpleNamespace(_pubs=[pub], _param_tables=[table], _backend=backend)
        QMSamplerJob._push_parameters(job, None, [0])
    else:
        job = SimpleNamespace(_backend=backend)
        plan = SimpleNamespace(pub=pub, param_table=table, observables_var=SimpleNamespace(input_type=None))
        QMEstimatorJob._push_plan_data(job, None, plan)
    expected = {"a": 0.1, "theta": [0.0, 0.0, 0.2, 0.0]} if array_mode else {
        "a": 0.1, "_theta_0_": 0.0, "_theta_1_": 0.0, "_theta_2_": 0.2, "_theta_3_": 0.0,
    }
    assert pushed == [expected]


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
def test_vector_pub_compile_time_assignment(array_mode, primitive, monkeypatch):
    from qiskit_qm_provider.job import qua_programs

    pub = _mixed_vector_pub()
    table = ParameterTable.from_qiskit(pub.circuit)
    monkeypatch.setattr(qua_programs, "_process_circuit", lambda *args, **kwargs: None)
    monkeypatch.setattr(qua_programs, "_process_observables_with_circuit", lambda *args, **kwargs: None)
    with qua.program() as prog:
        if primitive == "sampler":
            qua_programs._process_sampler_pub(pub, None, table)
        else:
            plan = SimpleNamespace(
                pub=pub, param_table=table,
                observables_var=SimpleNamespace(input_type=InputType.INPUT_STREAM),
            )
            qua_programs._process_estimator_pub(plan, None)
    assert "assign(" in generate_qua_script(prog)


def test_array_mode_input_name_collision_rejected(monkeypatch):
    monkeypatch.setattr(qiskit_classical, "qiskit_has_array", lambda: True)
    qc, _ = _partial_vector_circuit(name="theta")
    qc.rz(QiskitParameter("theta"), 0)
    with pytest.raises(ValueError, match="Ambiguous Qiskit input name 'theta'"):
        ParameterTable.from_qiskit(qc)
    compiler = _RecordingCompiler()
    with pytest.raises(ValueError, match="Ambiguous Qiskit input name"):
        _compile(qc, None, compiler)
    assert not compiler.calls


def test_array_mode_dictionary_vector_binding(monkeypatch):
    monkeypatch.setattr(qiskit_classical, "qiskit_has_array", lambda: True)
    qc, vec = _partial_vector_circuit()
    compiler = _RecordingCompiler()
    with qua.program():
        source = qua.declare(qua.fixed, value=[0.0] * len(vec))
        _compile(qc, {vec: source}, compiler)
        assert compiler.calls[-1].inputs == {vec.name: source}
        with pytest.raises(ValueError, match="whole ParameterVector"):
            _compile(qc, {vec[2]: source[2]}, compiler)


def test_singleton_row_copy():
    from qiskit_qm_provider import Parameter

    p = Parameter("one", [0.0])
    grid = QUA2DArray("grid", [[0.1], [0.2]])
    with qua.program() as prog:
        grid.declare()
        p.assign(grid[1])
    assert "for_(" in generate_qua_script(prog)
