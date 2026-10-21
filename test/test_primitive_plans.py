"""Primitive plans: packed observable rows, generated QUA shape, push order and IQCC gating."""

from __future__ import annotations

import random
import re
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Parameter
from qiskit.circuit.classical import types
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp
from qm import generate_qua_script, qua

from qiskit_qm_provider.job import qua_programs
from qiskit_qm_provider.job.primitive_plans import EstimatorPlan
from qiskit_qm_provider.job.pub_parameters import (
    OBS_QUBITS_PER_WORD,
    obs_words,
    pack_obs_codes,
    pack_obs_rows,
    unpack_obs_codes,
)
from qiskit_qm_provider.parameter_table import InputType, ParameterPool
from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions


@pytest.fixture(autouse=True)
def reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


def _estimator_plan(observables, values=None, input_type=InputType.INPUT_STREAM, num_qubits=2):
    qc = QuantumCircuit(num_qubits, name="plan")
    if values is not None:
        qc.rx(Parameter("theta"), 0)
    qc.add_register(ClassicalRegister(num_qubits, "__c"))
    for j in range(num_qubits):
        qc.add_input(f"obs_{j}", types.Uint(4))
    qc.measure(range(num_qubits), range(num_qubits))
    pub = EstimatorPub.coerce((qc, observables, values) if values is not None else (qc, observables), precision=0.1)
    return EstimatorPlan.from_pub(pub, QMEstimatorOptions(input_type=input_type), plan_index=0)


def _qua_script(plan) -> str:
    with qua.program() as prog:
        qua_programs._process_estimator_pub(plan, None)
    return generate_qua_script(prog)


@pytest.fixture
def no_circuit(monkeypatch):
    # The circuit body is irrelevant here; only input loading is inspected.
    monkeypatch.setattr(qua_programs, "_process_circuit", lambda *args, **kwargs: None)


# ---------------------------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_qubits", [1, 14, 15, 16, 30, 31, 45])
def test_pack_round_trip_at_word_boundaries(num_qubits):
    rng = random.Random(num_qubits)
    for _ in range(25):
        codes = [rng.randint(0, 2) for _ in range(num_qubits)]
        words = pack_obs_codes(codes)
        assert len(words) == obs_words(num_qubits)
        assert all(0 <= w < 2**30 for w in words)  # sign bit never used
        assert unpack_obs_codes(words, num_qubits) == codes


def test_obs_rows_are_padded_and_counted():
    rows = pack_obs_rows([[(1, 2, 0)], [(0, 0, 1), (2, 2, 2)]], 3)
    assert rows == [[1, 1 | (2 << 2), 0], [2, 1 << 4, 2 | (2 << 2) | (2 << 4)]]


# ---------------------------------------------------------------------------------------------
# Generated QUA
# ---------------------------------------------------------------------------------------------


def test_streamed_obs_row_one_advance_per_binding(no_circuit):
    n = 20  # 2 words per observable
    plan = _estimator_plan(
        [SparsePauliOp("X" * n), SparsePauliOp(["Z" * n, "Y" * n])], values=[[0.1], [0.2]], num_qubits=n
    )
    assert plan.words_per_obs == 2 and not plan.shared_obs_row
    script = _qua_script(plan)
    # Inside the binding loop: one advance for the parameter row, one for the observable row.
    assert script.count("advance_input_stream(") == 2
    assert "obs_row_0" in script
    assert "declare_input_stream(int, 'obs_0'" not in script and "'obs_0'" not in script  # no per-qubit streams
    # Each observable is unpacked with shifts/masks into the 20 obs inputs.
    assert script.count("&3)") == n


def test_shared_obs_row_received_once_before_loop(no_circuit):
    plan = _estimator_plan([SparsePauliOp(["ZX", "XZ"])], values=[[0.1], [0.2], [0.3]])
    assert plan.shared_obs_row
    script = _qua_script(plan)
    lines = script.splitlines()
    obs_advance = next(i for i, l in enumerate(lines) if "advance_input_stream" in l and "obs_row_0" in l)
    binding_loop = next(i for i, l in enumerate(lines) if l.strip().startswith("with for_(") and "<3)" in l)
    assert obs_advance < binding_loop


def test_compile_time_obs_rows_preloaded(no_circuit):
    plan = _estimator_plan([SparsePauliOp("XY"), SparsePauliOp("ZZ")], values=[[0.1], [0.2]], input_type=None)
    script = _qua_script(plan)
    assert "advance_input_stream" not in script and "pause()" not in script
    assert "&3)" in script


def test_io_obs_row_receives_only_used_prefix(no_circuit):
    plan = _estimator_plan([SparsePauliOp("XY"), SparsePauliOp(["ZZ", "XX"])], values=[[0.1], [0.2]], input_type=InputType.IO1)
    script = _qua_script(plan)
    # One pause for the count, then a word loop bounded by 1 + n_obs * words_per_obs (W = 1 here).
    assert re.search(r"<\(1\+\(a\d+\[0\]\*1\)\)\)", script)
    assert script.count("pause()") == 3  # parameter row (1 value) + count + one word per loop iteration


def test_opnic_observables_not_supported():
    with pytest.raises(NotImplementedError, match="OPNIC"):
        _estimator_plan([SparsePauliOp("ZX")], input_type=InputType.OPNIC)


# ---------------------------------------------------------------------------------------------
# Host push order and IQCC gating
# ---------------------------------------------------------------------------------------------


def test_local_push_matches_sync_hook_order(monkeypatch):
    plan = _estimator_plan([SparsePauliOp("XY"), SparsePauliOp(["ZZ", "XX"])], values=[[0.1], [0.2]])
    pushed = []
    monkeypatch.setattr(plan.param_table, "push_to_opx", lambda values, *a: pushed.append(("params", values)))
    monkeypatch.setattr(plan.obs_row, "push_to_opx", lambda values, *a: pushed.append(("obs", values)))
    plan.push(None)
    hook = [(name, value) for _, name, _, value in plan.sync_hook_pushes()]
    assert [v for _, v in pushed] == [
        {"param_table_0_row": [0.1]},
        plan.obs_rows[0],
        {"param_table_0_row": [0.2]},
        plan.obs_rows[1],
    ]
    assert [name for name, _ in hook] == ["param_table_0_row", "obs_row_0"] * 2


def test_iqcc_estimator_attaches_hook_for_observables_only():
    from qiskit_qm_provider.job.qm_estimator_job import IQCCEstimatorJob

    plan = _estimator_plan([SparsePauliOp("ZX")])  # no circuit parameters, streamed observables
    assert not plan.streams_params and plan.streams_inputs
    executed = []
    job = object.__new__(IQCCEstimatorJob)
    job._qm_jobs = None
    job._programs = ["program"]
    job._chunk_layout = [[0]]
    job._plans = [plan]
    job.metadata = {}
    job._backend = SimpleNamespace(
        qm=SimpleNamespace(execute=lambda prog, options: executed.append(dict(options)) or SimpleNamespace(id="1"))
    )
    job.submit()
    assert "sync_hook" in executed[0]
