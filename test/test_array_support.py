"""Tests for Qiskit array support (``arthurostrauss/qiskit@full_real_time_expr``).

Both Qiskit variants are covered: table logic runs in both layouts by monkeypatching
:func:`qiskit_supports_vector_arrays`; tests that need the fork's exporter feature / ``types.Array``
are skipped on stock Qiskit, and tests that need the stock per-element exporter are skipped on the
fork. Input binding is checked against what the exporter really declares, including renames.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit import Parameter as QiskitParameter, ParameterVector
from qiskit.circuit.classical import expr, types
from qm import qua, generate_qua_script

from qiskit_qm_provider import InputType, ParameterPool, ParameterTable, QUA2DArray
from qiskit_qm_provider.parameter_table import qiskit_classical
from qiskit_qm_provider.parameter_table.qiskit_classical import (
    bind_exported_inputs,
    qiskit_has_array,
    qiskit_supports_vector_arrays,
)
from qiskit_qm_provider.qasm3_exporter import ExportedInput, QMOpenQASM3Exporter

HAS_ARRAY = hasattr(types, "Array")
HAS_VECTOR_ARRAYS = hasattr(qasm3.ExperimentalFeatures, "PARAMETER_VECTORS_AS_ARRAYS")
fork_only = pytest.mark.skipif(not HAS_ARRAY, reason="requires Qiskit with types.Array (full_real_time_expr)")
vector_arrays_only = pytest.mark.skipif(
    not HAS_VECTOR_ARRAYS, reason="requires the PARAMETER_VECTORS_AS_ARRAYS exporter feature (full_real_time_expr)"
)
stock_only = pytest.mark.skipif(HAS_VECTOR_ARRAYS, reason="requires stock Qiskit exporter (per-element inputs)")
BASIS = ["rx", "rz", "x"]


@pytest.fixture(autouse=True)
def reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


@pytest.fixture(params=[True, False], ids=["array_mode", "per_element_mode"])
def array_mode(request, monkeypatch):
    monkeypatch.setattr(qiskit_classical, "qiskit_supports_vector_arrays", lambda: request.param)
    return request.param


def _export(qc, vector_arrays):
    """Export like the backend does, with or without vector arrays."""
    experimental = qasm3.ExperimentalFeatures(0)
    if vector_arrays:
        experimental = qasm3.ExperimentalFeatures.PARAMETER_VECTORS_AS_ARRAYS
    exporter = QMOpenQASM3Exporter(includes=(), basis_gates=BASIS, disable_constants=True, experimental=experimental)
    return exporter.export(qc)


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
    assert qiskit_supports_vector_arrays() is HAS_VECTOR_ARRAYS


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
    try:
        qc.add_input("arr", types.Array(types.Float(), 0))
    except ValueError:
        pytest.skip("this Qiskit rejects zero-size arrays itself (OpenQASM 3 sizes are positive)")
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


@vector_arrays_only
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


@vector_arrays_only
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


def _plan(primitive, pub, circuit_table, delivery_table):
    from qiskit_qm_provider.job.primitive_plans import EstimatorPlan, SamplerPlan

    return (SamplerPlan if primitive == "sampler" else EstimatorPlan)(pub, 0, circuit_table, delivery_table)


def _push_with(primitive, pub, circuit_table, delivery_table):
    _plan(primitive, pub, circuit_table, delivery_table).push(None)


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
def test_vector_pub_dense_row_streaming(array_mode, primitive, monkeypatch):
    from qiskit_qm_provider.job.pub_parameters import build_pub_tables

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
    from qiskit_qm_provider.job.pub_parameters import build_pub_tables

    pub = _mixed_vector_pub()
    circuit_table, delivery_table = build_pub_tables(pub.circuit, input_type, "pt")
    monkeypatch.setattr(qua_programs, "_process_circuit", lambda *args, **kwargs: None)
    plan = _plan(primitive, pub, circuit_table, delivery_table)
    if primitive == "estimator":  # only parameter scatter is inspected here
        monkeypatch.setattr(plan, "declare_observables", lambda: None)
        monkeypatch.setattr(plan, "observable_groups", lambda p: nullcontext())
        monkeypatch.setattr(type(plan), "shots", 1)  # the PUB here is a SamplerPub (no precision)
    with qua.program() as prog:
        if primitive == "sampler":
            qua_programs._process_sampler_pub(plan, None)
        else:
            qua_programs._process_estimator_pub(plan, None)
    script = generate_qua_script(prog)
    # Only the two bound entries are assigned per binding; unused vector elements are not rewritten.
    assert script.count("assign(") == 2
    if input_type is not None:
        assert "advance_input_stream" in script and "pt_row" in script


def test_array_mode_input_name_collision_rejected(monkeypatch, compile_circuit):
    monkeypatch.setattr(qiskit_classical, "qiskit_supports_vector_arrays", lambda: True)
    qc, _ = _partial_vector_circuit(name="theta")
    qc.rz(QiskitParameter("theta"), 0)
    with pytest.raises(ValueError, match="Ambiguous Qiskit input name 'theta'"):
        ParameterTable.from_qiskit(qc)
    compiler = _RecordingCompiler()
    with pytest.raises(ValueError, match="Ambiguous Qiskit input name"):
        compile_circuit(qc, None, compiler)
    assert not compiler.calls


def test_dictionary_vector_binding(compile_circuit):
    qc, vec = _partial_vector_circuit()
    compiler = _RecordingCompiler()
    with qua.program():
        source = qua.declare(qua.fixed, value=[0.0] * len(vec))
        compile_circuit(qc, {vec: source}, compiler)
        (name, value), = compiler.calls[-1].inputs.items()
        if HAS_VECTOR_ARRAYS:  # exported as one array: the vector binds it whole
            assert name == "t" and value is source
        else:  # exported per element: the vector binds the used element's cell
            assert name == "_t_2_" and str(value).endswith("[2]")
        element = qua.declare(qua.fixed)
        if HAS_VECTOR_ARRAYS:
            with pytest.raises(ValueError, match="bind the whole vector"):
                compile_circuit(qc, {vec[2]: element}, compiler)
        else:
            compile_circuit(qc, {vec[2]: element}, compiler)
            assert compiler.calls[-1].inputs["_t_2_"] is element


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


def test_dictionary_string_keys_follow_object_rules(compile_circuit):
    qc, vec = _partial_vector_circuit()
    compiler = _RecordingCompiler()
    with qua.program():
        source = qua.declare(qua.fixed, value=[0.0] * len(vec))
        compile_circuit(qc, {"t": source}, compiler)
        assert list(compiler.calls[-1].inputs) == (["t"] if HAS_VECTOR_ARRAYS else ["_t_2_"])
        element = qua.declare(qua.fixed)
        if HAS_VECTOR_ARRAYS:
            with pytest.raises(ValueError, match="bind the whole vector"):
                compile_circuit(qc, {"t[2]": element}, compiler)
        else:
            compile_circuit(qc, {"t[2]": element}, compiler)
            assert list(compiler.calls[-1].inputs) == ["_t_2_"]


@vector_arrays_only
def test_compile_binds_renamed_vector_input(compile_circuit):
    # `x` is a basis gate, so the exporter renames the vector array; the table field keeps `x`.
    qc = QuantumCircuit(1)
    vec = ParameterVector("x", 2)
    qc.rx(vec[0], 0)
    table = ParameterTable.from_qiskit(qc)
    compiler = _RecordingCompiler()
    with qua.program():
        table.declare(pause_program=False)
        compile_circuit(qc, table, compiler)
        inputs = compiler.calls[0].inputs
        assert list(inputs) == ["x_0"] and inputs["x_0"] is table.get_parameter("x").var
    assert "input array[float[64], 2] x_0;" in compiler.calls[0].qasm


def test_unmarked_table_is_not_a_row(monkeypatch):
    # A separately built table that is not the circuit table must not be mistaken for a dense row.
    from qiskit_qm_provider import Parameter
    from qiskit_qm_provider.job.pub_parameters import PubRowTable, build_pub_tables, is_row_delivery

    pub = _mixed_vector_pub()
    circuit_table, row_table = build_pub_tables(pub.circuit, InputType.INPUT_STREAM, "pt")
    assert isinstance(row_table, PubRowTable) and row_table.row.name == "pt_row"
    assert is_row_delivery(row_table) and not is_row_delivery(circuit_table)
    other = ParameterTable.from_qiskit(pub.circuit, input_type=InputType.INPUT_STREAM, name="other")
    assert not is_row_delivery(other)
    # A plain table shaped and named like a row table is still not one: the type decides.
    lookalike = ParameterTable([Parameter("other_row", [0.0] * 2, input_type=InputType.INPUT_STREAM)], name="other_row")
    assert not is_row_delivery(lookalike)
    pushed = []
    monkeypatch.setattr(other, "push_to_opx", lambda values, *args: pushed.append(values))
    _push_with("sampler", pub, circuit_table, other)
    assert set(pushed[0]) == set(other.table)  # packed into its own fields, not {first_field: row}


def test_row_table_survives_deepcopy():
    import copy

    from qiskit_qm_provider.job.pub_parameters import PubRowTable, build_pub_tables, delivery_rows, is_row_delivery

    pub = _mixed_vector_pub()
    circuit_table, row_table = build_pub_tables(pub.circuit, InputType.INPUT_STREAM, "pt")
    copied = copy.deepcopy(row_table)
    assert copied is not row_table and isinstance(copied, PubRowTable) and is_row_delivery(copied)
    assert copied.row is not row_table.row and copied.row.length == row_table.row.length
    # The copy still delivers raw rows (not zero-filled packed fields).
    assert delivery_rows(pub, circuit_table, copied) == delivery_rows(pub, circuit_table, row_table) == [[[0.1, 0.2]]]


def test_row_table_rejects_mismatched_circuit():
    from qiskit_qm_provider.job.pub_parameters import build_pub_tables, delivery_rows

    pub = _mixed_vector_pub()
    _, row_table = build_pub_tables(pub.circuit, InputType.INPUT_STREAM, "pt")
    with pytest.raises(ValueError, match="does not match the circuit"):
        delivery_rows(pub, row_table, row_table)


def test_estimator_row_tables_named_by_plan_index():
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit import ClassicalRegister
    from qiskit_qm_provider.job.primitive_plans import EstimatorPlan
    from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions

    plans = []
    for i, size in enumerate((1, 2)):
        qc = QuantumCircuit(1, name="ansatz")  # same name, different parameter counts
        for p in ParameterVector(f"v{i}", size):
            qc.rx(p, 0)
        qc.add_register(ClassicalRegister(1, "__c"))
        qc.add_input("__obs_0", types.Uint(4))
        qc.measure(0, 0)
        pub = EstimatorPub.coerce((qc, SparsePauliOp(["Z"]), [[0.1] * size]))
        plans.append(EstimatorPlan.from_pub(pub, QMEstimatorOptions(input_type=InputType.INPUT_STREAM), plan_index=i))
    assert [list(plan.param_table.table) for plan in plans] == [["param_table_0_row"], ["param_table_1_row"]]


# ---------------------------------------------------------------------------------------------
# Input binding against the exporter's real declarations (no QuAM needed)
# ---------------------------------------------------------------------------------------------


def _vector_named_like_a_gate():
    qc = QuantumCircuit(1)
    vec = ParameterVector("x", 2)  # `x` is also a basis gate (e.g. zz_feature_map's vector)
    qc.rx(vec[1], 0)
    qc.x(0)
    return qc, vec


@vector_arrays_only
def test_exported_inputs_report_renamed_vector_array():
    qc, vec = _vector_named_like_a_gate()
    source, exported = _export(qc, vector_arrays=True)
    assert "input array[float[64], 2] x_0;" in source and "rx(x_0[1])" in source
    assert exported == {vec: ExportedInput("x_0", 2)}


def test_exported_inputs_report_scalar_elements():
    qc, vec = _vector_named_like_a_gate()
    scalar = QiskitParameter("theta.1")  # not an identifier: the exporter escapes it
    qc.rz(scalar, 0)
    source, exported = _export(qc, vector_arrays=False)
    assert exported == {vec[1]: ExportedInput("_x_1_"), scalar: ExportedInput("_theta_1")}
    assert "input float[64] _theta_1;" in source


@pytest.mark.parametrize("vector_arrays", [True, False], ids=["array_export", "scalar_export"])
def test_bind_follows_exporter_renames(vector_arrays):
    if vector_arrays and not HAS_VECTOR_ARRAYS:
        pytest.skip("requires the PARAMETER_VECTORS_AS_ARRAYS exporter feature")
    qc, vec = _vector_named_like_a_gate()
    scalar = QiskitParameter("theta.1")
    qc.rz(scalar, 0)
    _, exported = _export(qc, vector_arrays)
    with qua.program():
        array, angle = qua.declare(qua.fixed, value=[0.0, 0.0]), qua.declare(qua.fixed)
        inputs = bind_exported_inputs(exported, {"x": array, "theta.1": angle})
    assert inputs["_theta_1"] is angle
    if vector_arrays:
        assert set(inputs) == {"x_0", "_theta_1"} and inputs["x_0"] is array
    else:  # an array field binds the element's scalar input through its cell
        assert set(inputs) == {"_x_1_", "_theta_1"} and str(inputs["_x_1_"]).endswith("[1]")


@vector_arrays_only
def test_bind_rejects_per_element_fields_for_array_export():
    qc, vec = _vector_named_like_a_gate()
    _, exported = _export(qc, vector_arrays=True)
    with qua.program():
        element = qua.declare(qua.fixed)
        with pytest.raises(ValueError, match="'x_0'.*has no binding.*bind the whole vector"):
            bind_exported_inputs(exported, {"_x_1_": element})


@vector_arrays_only
def test_bind_rejects_wrong_shapes():
    # qm-qasm would broadcast a scalar into every element, or fail opaquely on a length mismatch.
    qc, _ = _vector_named_like_a_gate()
    _, exported = _export(qc, vector_arrays=True)
    with qua.program():
        scalar, short = qua.declare(qua.fixed), qua.declare(qua.fixed, value=[0.0])
        with pytest.raises(ValueError, match="is bound to a scalar"):
            bind_exported_inputs(exported, {"x": scalar})
        with pytest.raises(ValueError, match="is bound to an array of length 1"):
            bind_exported_inputs(exported, {"x": short})


@fork_only
def test_bind_checks_classical_array_inputs():
    qc = QuantumCircuit(1)
    ops = qc.add_input("x", types.Array(types.Uint(8), 3))  # renamed next to the `x` gate
    with qc.if_test(expr.equal(expr.index(ops, 0), 1)):
        qc.x(0)
    _, exported = _export(qc, vector_arrays=True)
    assert exported == {ops: ExportedInput("x_0", 3)}
    with qua.program():
        good, bad = qua.declare(int, value=[0] * 3), qua.declare(int, value=[0] * 2)
        assert bind_exported_inputs(exported, {"x": good}) == {"x_0": good}
        with pytest.raises(ValueError, match="is bound to an array of length 2"):
            bind_exported_inputs(exported, {"x": bad})


def test_bind_lists_missing_inputs_and_ignores_extra_fields():
    qc = QuantumCircuit(1)
    a, b = QiskitParameter("a"), QiskitParameter("b")
    qc.rx(a, 0)
    qc.rz(b, 0)
    _, exported = _export(qc, vector_arrays=False)
    with qua.program():
        value, unrelated = qua.declare(qua.fixed), qua.declare(qua.fixed)
        assert set(bind_exported_inputs(exported, {"a": value, "b": value, "other": unrelated})) == {"a", "b"}
        with pytest.raises(ValueError, match="Parameter 'b' \\(exported as 'b', a scalar\\) has no binding"):
            bind_exported_inputs(exported, {"a": value})


def _compile_with_qm_qasm(source, inputs):
    """Compile with the real qm-qasm compiler against a one-qubit hardware config."""
    from qm_qasm import Compiler, HardwareConfig, OperationIdentifier

    def rx(theta):
        qua.play("x90" * qua.amp(theta), "q0.xy")

    def x():
        qua.play("x180", "q0.xy")

    def rz(theta):
        qua.frame_rotation_2pi(theta, "q0.xy")

    operations = {
        OperationIdentifier("rx", 1, (0,)): rx,
        OperationIdentifier("rz", 1, (0,)): rz,
        OperationIdentifier("x", 0, (0,)): x,
    }
    config = HardwareConfig(quantum_operations_db=operations, physical_qubits={0: ("q0.xy",)})
    compiler = Compiler(hardware_config=config)
    return compiler.compile(source, inputs=inputs)


@pytest.mark.parametrize("vector_arrays", [True, False], ids=["array_export", "scalar_export"])
def test_renamed_inputs_compile_with_qm_qasm(vector_arrays):
    # Before binding by object, a vector named like a gate (array export) and a parameter whose
    # name is not an identifier (both exports) failed with "Pass Qua1GenerationPass failed".
    if vector_arrays and not HAS_VECTOR_ARRAYS:
        pytest.skip("requires the PARAMETER_VECTORS_AS_ARRAYS exporter feature")
    qc, vec = _vector_named_like_a_gate()
    qc.rz(QiskitParameter("theta.1"), 0)
    physical = transpile(qc, basis_gates=BASIS, initial_layout=[0], optimization_level=0)
    table = ParameterTable.from_qiskit(physical, name="renamed")
    if not qiskit_supports_vector_arrays():
        assert set(table.table) == {"_x_1_", "theta.1"}
    source, exported = _export(physical, vector_arrays)
    with qua.program() as prog:
        table.declare(pause_program=False)
        _compile_with_qm_qasm(source, bind_exported_inputs(exported, table.variables_dict))
    script = generate_qua_script(prog)
    assert "amplitude_scale" in script and "frame_rotation_2pi" in script



@vector_arrays_only
def test_backend_option_exports_vectors_as_scalars(compilation_backend, compile_circuit):
    # With the option off, the branch exports used elements as scalar inputs, and the
    # vector's array field binds them through its cells, with the real compiler too.
    qc, _ = _partial_vector_circuit(name="theta")
    table = ParameterTable.from_qiskit(qc)
    compiler = _RecordingCompiler()
    compilation_backend.set_options(export_vector_as_array=False)
    try:
        with qua.program() as prog:
            table.declare(pause_program=False)
            compile_circuit(qc, table, compiler)
            inputs = compiler.calls[-1].inputs
            assert list(inputs) == ["_theta_2_"] and str(inputs["_theta_2_"]).endswith("[2]")
            compile_circuit(qc, table)
    finally:
        compilation_backend.set_options(export_vector_as_array=True)
    assert "input float[64] _theta_2_;" in compiler.calls[-1].qasm
    assert "input array" not in compiler.calls[-1].qasm
    assert "[2]" in generate_qua_script(prog)
