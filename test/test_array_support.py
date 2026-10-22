"""Tests for Qiskit array support (``arthurostrauss/qiskit@full_real_time_expr``).

Both Qiskit variants are covered: table logic runs in both modes by monkeypatching
:func:`qiskit_has_array`; tests that need the fork's exporter / ``types.Array`` are skipped on
stock Qiskit, and tests that need the stock per-element exporter are skipped on the fork.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter as QiskitParameter, ParameterVector
from qiskit.circuit.classical import expr, types
from qiskit.qasm3 import Exporter
from qm import qua, generate_qua_script

from qiskit_qm_provider import InputType, ParameterPool, ParameterTable, QUA2DArray
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


@pytest.fixture
def compile_circuit(compilation_backend, monkeypatch):
    """Compile through a real machine-derived target; replace only the compiler when
    inspecting input mappings that deliberately emulate the other Qiskit variant.
    """
    def compile(qc, table, compiler=None):
        physical = transpile(qc, compilation_backend, optimization_level=0)
        with monkeypatch.context() as patch:
            if compiler is not None:
                patch.setattr(type(compilation_backend), "compiler", property(lambda self: compiler))
            return compilation_backend.quantum_circuit_to_qua(physical, table)

    return compile


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
        assert list(table.table) == ["_t_2_"]  # stock exporter emits only used elements


def test_filter_split_vector(array_mode):
    qc = QuantumCircuit(1)
    vec = ParameterVector("t", 3)
    for p in vec:
        qc.rx(p, 0)
    keep_first = lambda p: getattr(p, "index", None) == 0  # noqa: E731
    table = ParameterTable.from_qiskit(qc, filter_function=keep_first)
    # array mode: the full vector is declared; stock: only the kept element
    assert list(table.table) == (["t"] if array_mode else ["_t_0_"])


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
def test_duration_stretch_input_rejected(make_type, compile_circuit):
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
        compile_circuit(qc, None, _RecordingCompiler())


def test_duration_declared_var_rejected_in_compile(compile_circuit):
    qc = QuantumCircuit(1)
    qc.add_var("d", expr.Value(10, types.Duration()))
    with pytest.raises(ValueError, match="Unsupported classical type"):
        compile_circuit(qc, None, _RecordingCompiler())


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
def test_zero_size_array_rejected(compile_circuit):
    qc = QuantumCircuit(1)
    qc.add_input("arr", types.Array(types.Float(), 0))
    with pytest.raises(ValueError, match="zero-size"):
        ParameterTable.from_qiskit(qc)
    with pytest.raises(ValueError, match="zero-size"):
        compile_circuit(qc, None, _RecordingCompiler())


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
def test_e2e_fork_partial_vector_with_qua2d_row(compile_circuit):
    qc, vec = _partial_vector_circuit(name="theta")
    table = ParameterTable.from_qiskit(qc)
    grid = QUA2DArray("grid", [[0.1] * 5, [0.2] * 5])
    compiler = _RecordingCompiler()
    with qua.program() as prog:
        table.declare(pause_program=False)
        grid.declare()
        table.get_parameter("theta").assign(grid[1])
        compile_circuit(qc, table, compiler)
        assert compiler.calls[0].inputs["theta"] is table.get_parameter("theta").var
    call = compiler.calls[0]
    assert "input array[float[64], 5] theta;" in call.qasm
    assert list(call.inputs) == ["theta"]
    assert "for_(" in generate_qua_script(prog)


@fork_only
def test_e2e_fork_real_compiler(compile_circuit):
    from packaging.version import Version
    from importlib.metadata import version

    if Version(version("qm-qasm")) < Version("1.8.0"):
        pytest.skip("array inputs need qm-qasm >= 1.8.0")
    qc, _ = _partial_vector_circuit()
    table = ParameterTable.from_qiskit(qc)
    with qua.program():
        compile_circuit(qc, table)


@stock_only
def test_e2e_stock_real_compiler(compile_circuit):
    qc, _ = _partial_vector_circuit()
    qc.rz(QiskitParameter("a"), 0)
    table = ParameterTable.from_qiskit(qc)
    with qua.program() as prog:
        compile_circuit(qc, table)
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


def _push_with(primitive, pub, circuit_table, delivery_table):
    from qiskit_qm_provider.job.qm_sampler_job import QMSamplerJob
    from qiskit_qm_provider.job.qm_estimator_job import QMEstimatorJob

    backend = SimpleNamespace(qm=None)
    if primitive == "sampler":
        job = SimpleNamespace(
            _pubs=[pub], _param_tables=[delivery_table], _circuit_tables=[circuit_table], _backend=backend
        )
        QMSamplerJob._push_parameters(job, None, [0])
    else:
        job = SimpleNamespace(_backend=backend)
        plan = SimpleNamespace(
            pub=pub, param_table=delivery_table, circuit_table=circuit_table,
            observables_var=SimpleNamespace(input_type=None),
        )
        QMEstimatorJob._push_plan_data(job, None, plan)


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
def test_vector_pub_dense_row_streaming(array_mode, primitive, monkeypatch):
    from qiskit_qm_provider.job._pub_parameters import build_pub_tables

    pub = _mixed_vector_pub()
    circuit_table, delivery_table = build_pub_tables(pub.circuit, InputType.INPUT_STREAM, "pt")
    assert circuit_table.input_type is None and list(delivery_table.table) == ["pt_row"]
    pushed = []
    monkeypatch.setattr(delivery_table, "push_to_opx", lambda values, *args: pushed.append(values))
    _push_with(primitive, pub, circuit_table, delivery_table)
    # Raw row in circuit-parameter order (a, theta[2]), independent of the vector layout.
    assert pushed == [{"pt_row": [0.1, 0.2]}]


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
def test_vector_pub_packed_streaming(array_mode, primitive, monkeypatch):
    # A table that delivers its own fields (OPNIC-style) still receives packed values.
    pub = _mixed_vector_pub()
    table = ParameterTable.from_qiskit(pub.circuit, input_type=InputType.INPUT_STREAM)
    pushed = []
    monkeypatch.setattr(table, "push_to_opx", lambda values, *args: pushed.append(values))
    _push_with(primitive, pub, table, table)
    expected = {"a": 0.1, "theta": [0.0, 0.0, 0.2, 0.0]} if array_mode else {"a": 0.1, "_theta_2_": 0.2}
    assert pushed == [expected]


@pytest.mark.parametrize("input_type", [None, InputType.INPUT_STREAM], ids=["compile_time", "dense_row"])
@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
def test_vector_pub_qua_scatter(array_mode, primitive, input_type, monkeypatch):
    from qiskit_qm_provider.job import qua_programs
    from qiskit_qm_provider.job._pub_parameters import build_pub_tables

    pub = _mixed_vector_pub()
    circuit_table, delivery_table = build_pub_tables(pub.circuit, input_type, "pt")
    monkeypatch.setattr(qua_programs, "_process_circuit", lambda *args, **kwargs: None)
    monkeypatch.setattr(qua_programs, "_process_observables_with_circuit", lambda *args, **kwargs: None)
    with qua.program() as prog:
        if primitive == "sampler":
            qua_programs._process_sampler_pub(pub, None, delivery_table, circuit_table)
        else:
            plan = SimpleNamespace(
                pub=pub, param_table=delivery_table, circuit_table=circuit_table,
                observables_var=SimpleNamespace(input_type=InputType.INPUT_STREAM),
            )
            qua_programs._process_estimator_pub(plan, None)
    script = generate_qua_script(prog)
    # Only the two bound entries are assigned per binding; unused vector elements are not rewritten.
    assert script.count("assign(") == 2
    if input_type is not None:
        assert "advance_input_stream" in script and "pt_row" in script


def test_array_mode_input_name_collision_rejected(monkeypatch, compile_circuit):
    monkeypatch.setattr(qiskit_classical, "qiskit_has_array", lambda: True)
    qc, _ = _partial_vector_circuit(name="theta")
    qc.rz(QiskitParameter("theta"), 0)
    with pytest.raises(ValueError, match="Ambiguous Qiskit input name 'theta'"):
        ParameterTable.from_qiskit(qc)
    compiler = _RecordingCompiler()
    with pytest.raises(ValueError, match="Ambiguous Qiskit input name"):
        compile_circuit(qc, None, compiler)
    assert not compiler.calls


def test_array_mode_dictionary_vector_binding(monkeypatch, compile_circuit):
    monkeypatch.setattr(qiskit_classical, "qiskit_has_array", lambda: True)
    qc, vec = _partial_vector_circuit()
    compiler = _RecordingCompiler()
    with qua.program():
        source = qua.declare(qua.fixed, value=[0.0] * len(vec))
        compile_circuit(qc, {vec: source}, compiler)
        assert compiler.calls[-1].inputs == {vec.name: source}
        with pytest.raises(ValueError, match="whole ParameterVector"):
            compile_circuit(qc, {vec[2]: source[2]}, compiler)


def test_singleton_row_copy():
    from qiskit_qm_provider import Parameter

    p = Parameter("one", [0.0])
    grid = QUA2DArray("grid", [[0.1], [0.2]])
    with qua.program() as prog:
        grid.declare()
        p.assign(grid[1])
    assert "for_(" in generate_qua_script(prog)


def test_from_qiskit_accepts_pulse_schedule(array_mode):
    pulse = pytest.importorskip("qiskit.pulse")
    amp = QiskitParameter("amp")
    with pulse.build(name="sched") as sched:
        pulse.play(pulse.Constant(16, amp), pulse.DriveChannel(0))
    table = ParameterTable.from_qiskit(sched)
    assert list(table.table) == ["amp"]


def test_packer_typed_zeros_and_shape_check(array_mode):
    from qiskit_qm_provider import Parameter
    from qiskit_qm_provider.parameter_table.qiskit_classical import qiskit_parameter_packer

    qc, vec = _partial_vector_circuit(name="t", size=3)
    flag = Parameter("flag", qua_type=bool)
    table = ParameterTable([*ParameterTable.from_qiskit(qc).parameters, flag])
    packed = qiskit_parameter_packer(qc, table)([0.5])
    assert packed["flag"] is False
    if array_mode:
        assert packed["t"] == [0.0, 0.0, 0.5]
        bad = ParameterTable([Parameter("t", 0.0)], name="bad")
        with pytest.raises(ValueError, match="must be an array of length 3"):
            qiskit_parameter_packer(qc, bad)
    else:
        assert packed["_t_2_"] == 0.5


def test_dictionary_string_keys_follow_object_rules(array_mode, compile_circuit):
    qc, vec = _partial_vector_circuit()
    compiler = _RecordingCompiler()
    with qua.program():
        if array_mode:
            source = qua.declare(qua.fixed, value=[0.0] * len(vec))
            compile_circuit(qc, {"t": source}, compiler)
            assert compiler.calls[-1].inputs == {"t": source}
            with pytest.raises(ValueError, match="whole ParameterVector"):
                compile_circuit(qc, {"t[2]": source[2]}, compiler)
        else:
            x = qua.declare(qua.fixed)
            compile_circuit(qc, {"t[2]": x}, compiler)
            assert compiler.calls[-1].inputs == {"_t_2_": x}
            with pytest.raises(ValueError, match="require Qiskit array support"):
                compile_circuit(qc, {"t": x}, compiler)


def test_unmarked_table_is_not_a_row(monkeypatch):
    # A separately built table that is not the circuit table must not be mistaken for a dense row.
    from qiskit_qm_provider.job._pub_parameters import build_pub_tables, is_row_delivery

    pub = _mixed_vector_pub()
    circuit_table, row_table = build_pub_tables(pub.circuit, InputType.INPUT_STREAM, "pt")
    assert is_row_delivery(row_table) and not is_row_delivery(circuit_table)
    other = ParameterTable.from_qiskit(pub.circuit, input_type=InputType.INPUT_STREAM, name="other")
    assert not is_row_delivery(other)
    pushed = []
    monkeypatch.setattr(other, "push_to_opx", lambda values, *args: pushed.append(values))
    _push_with("sampler", pub, circuit_table, other)
    assert set(pushed[0]) == set(other.table)  # packed into its own fields, not {first_field: row}


def test_row_table_rejects_mismatched_circuit():
    from qiskit_qm_provider.job._pub_parameters import build_pub_tables, push_pub_rows

    pub = _mixed_vector_pub()
    _, row_table = build_pub_tables(pub.circuit, InputType.INPUT_STREAM, "pt")
    with pytest.raises(ValueError, match="does not match the circuit"):
        push_pub_rows(pub, row_table, row_table, job=None)


def test_estimator_row_tables_named_by_plan_index():
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_qm_provider.job.qm_estimator_job import _ExecutionPlan
    from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions

    plans = []
    for i, size in enumerate((1, 2)):
        qc = QuantumCircuit(1, name="ansatz")  # same name, different parameter counts
        for p in ParameterVector(f"v{i}", size):
            qc.rx(p, 0)
        pub = EstimatorPub.coerce((qc, SparsePauliOp(["Z"]), [[0.1] * size]))
        plans.append(_ExecutionPlan.from_pub(pub, QMEstimatorOptions(input_type=InputType.INPUT_STREAM), plan_index=i))
    assert [list(plan.param_table.table) for plan in plans] == [["param_table_0_row"], ["param_table_1_row"]]
