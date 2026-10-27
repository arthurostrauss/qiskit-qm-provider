# Copyright 2026 Arthur Strauss
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for self-contained, Jinja-based sync-hook generation.

The generated hooks must run on the IQCC cloud side without importing
``qiskit_qm_provider`` or ``numpy``. These tests build small sampler/estimator plans,
render the hooks, and exec them against a stub job to verify the right QM-job calls
are made, in order, with the right coercions.
"""

import numpy as np
import pytest
from qm.qua import fixed
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Parameter
from qiskit.circuit.classical import types
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_qm_provider.parameter_table import InputType, ParameterPool, ParameterTable
from qiskit_qm_provider.job.sync_hook import generate_sync_hook
from qiskit_qm_provider.job.primitive_plans import EstimatorPlan, SamplerPlan
from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions

from ._obs_packing import unpack_obs_codes


@pytest.fixture(autouse=True)
def reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


class _FakePub:
    def __init__(self, values, names=("theta",)):
        self.circuit = QuantumCircuit(1)
        parameters = [Parameter(name) for name in names]
        for parameter in parameters:
            self.circuit.rx(parameter, 0)
        self.parameter_values = BindingsArray({tuple(parameters): np.asarray(values)})


def _table_plan(pub, table):
    """A sampler plan whose table both is read by the circuit and delivers (packed fields)."""
    return SamplerPlan(pub, 0, table, table)


def _estimator_plan(observables, values=None, input_type=InputType.INPUT_STREAM, num_qubits=2):
    """Estimator plan for a circuit shaped like a validated estimator PUB (``__c`` + ``obs_j``)."""
    qc = QuantumCircuit(num_qubits)
    theta = Parameter("theta")
    if values is not None:
        qc.rx(theta, 0)
    qc.add_register(ClassicalRegister(num_qubits, "__c"))
    for j in range(num_qubits):
        qc.add_input(f"obs_{j}", types.Uint(4))
    qc.measure(range(num_qubits), range(num_qubits))
    pub = EstimatorPub.coerce((qc, observables, values) if values is not None else (qc, observables))
    return EstimatorPlan.from_pub(pub, QMEstimatorOptions(input_type=input_type), plan_index=0)


class _StubJob:
    """Records the calls a sync hook makes so we can assert behavior."""

    def __init__(self):
        self.streamed = []  # (name, value)
        self.io_values = []  # dict
        self.resumed = 0

    def is_paused(self):
        return True

    def push_to_input_stream(self, name, value):
        self.streamed.append((name, value))

    def set_io_values(self, **kwargs):
        self.io_values.append(kwargs)

    def resume(self):
        self.resumed += 1


def _exec_hook(code, job):
    """Exec a generated sync hook with ``get_qm_job`` stubbed to return ``job``."""
    import types

    fake_runtime = types.ModuleType("iqcc_cloud_client.runtime")
    fake_runtime.get_qm_job = lambda: job
    fake_pkg = types.ModuleType("iqcc_cloud_client")
    fake_pkg.runtime = fake_runtime

    import sys

    saved = {k: sys.modules.get(k) for k in ("iqcc_cloud_client", "iqcc_cloud_client.runtime")}
    sys.modules["iqcc_cloud_client"] = fake_pkg
    sys.modules["iqcc_cloud_client.runtime"] = fake_runtime
    try:
        exec(compile(code, "<sync_hook>", "exec"), {})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _assert_no_provider_imports(code):
    assert "from iqcc_cloud_client.runtime import get_qm_job" in code
    assert "qiskit_qm_provider" not in code
    assert "import numpy" not in code
    assert "push_to_opx" not in code
    # Valid Python source.
    compile(code, "<sync_hook>", "exec")


def test_sampler_input_stream_pushes_with_coercion():
    table = ParameterTable(
        {"x": (0.0, fixed, InputType.INPUT_STREAM), "n": (0, int, InputType.INPUT_STREAM)},
        name="t",
    )
    pub = _FakePub([[0.1, 2.0], [0.3, 4.0]], names=("x", "n"))
    code = generate_sync_hook([_table_plan(pub, table)])
    _assert_no_provider_imports(code)

    job = _StubJob()
    _exec_hook(code, job)

    # Two parameter sets, two params each -> 4 stream pushes, with int/fixed coercion.
    assert job.streamed == [("x", 0.1), ("n", 2), ("x", 0.3), ("n", 4)]
    assert all(isinstance(v, float) for (n, v) in job.streamed if n == "x")
    assert all(isinstance(v, int) for (n, v) in job.streamed if n == "n")


def test_sampler_dense_row_one_push_per_binding():
    pub = _FakePub([[0.1, 2.0], [0.3, 4.0]], names=("x", "y"))
    code = generate_sync_hook([SamplerPlan.from_pub(pub, InputType.INPUT_STREAM, 0)])
    _assert_no_provider_imports(code)

    job = _StubJob()
    _exec_hook(code, job)

    # One push of the whole raw row per binding, in circuit-parameter order.
    assert job.streamed == [("param_table_0_row", [0.1, 2.0]), ("param_table_0_row", [0.3, 4.0])]


def test_sampler_dense_row_io_pushes_elementwise():
    pub = _FakePub([[0.5, 0.25]], names=("x", "y"))
    code = generate_sync_hook([SamplerPlan.from_pub(pub, InputType.IO1, 0)])

    job = _StubJob()
    _exec_hook(code, job)

    assert job.io_values == [{"io1": 0.5}, {"io1": 0.25}]
    assert job.resumed == 2


def test_sampler_io_pushes_pause_resume():
    table = ParameterTable({"a": (0.0, fixed, InputType.IO1)}, name="io")
    pub = _FakePub([[0.5]], names=("a",))
    code = generate_sync_hook([_table_plan(pub, table)])
    _assert_no_provider_imports(code)

    job = _StubJob()
    _exec_hook(code, job)

    assert job.io_values == [{"io1": 0.5}]
    assert job.resumed == 1


def test_sampler_none_table_pushes_nothing():
    pub = _FakePub([[0.1]])
    code = generate_sync_hook([SamplerPlan(pub, 0, None, None)])
    _assert_no_provider_imports(code)

    job = _StubJob()
    _exec_hook(code, job)
    assert job.streamed == []
    assert job.io_values == []


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
@pytest.mark.parametrize("input_type", [InputType.INPUT_STREAM, InputType.IO1])
def test_array_pub_sync_hook(primitive, input_type, monkeypatch):
    from qiskit.circuit import ParameterVector
    from qiskit_qm_provider.parameter_table import qiskit_classical

    monkeypatch.setattr(qiskit_classical, "qiskit_has_array", lambda: True)
    qc = QuantumCircuit(1)
    vec = ParameterVector("theta", 4)
    scalar = Parameter("a")
    qc.rx(vec[2], 0)
    qc.rz(scalar, 0)
    pub = _FakePub([[0.2, 0.1]])
    pub.circuit = qc
    pub.parameter_values = BindingsArray({(vec[2], scalar): [[0.2, 0.1]]})
    table = ParameterTable.from_qiskit(qc, input_type=input_type)
    plan_cls = SamplerPlan if primitive == "sampler" else EstimatorPlan
    code = generate_sync_hook([plan_cls(pub, 0, table, table)])
    job = _StubJob()
    _exec_hook(code, job)
    if input_type == InputType.INPUT_STREAM:
        assert job.streamed == [("a", 0.1), ("theta", [0.0, 0.0, 0.2, 0.0])]
    else:
        assert job.io_values == [{"io1": value} for value in [0.1, 0.0, 0.0, 0.2, 0.0]]
        assert job.resumed == 5


def test_estimator_pushes_param_row_then_packed_obs_row():
    # Binding 0 measures XY (one group); binding 1 measures ZZ and XX (two groups).
    plan = _estimator_plan([SparsePauliOp("XY"), SparsePauliOp(["ZZ", "XX"])], values=[[0.1], [0.2]])
    assert plan.obs_codes[0] == [(2, 1)]  # qubit 0: Y -> 2, qubit 1: X -> 1
    assert sorted(plan.obs_codes[1]) == [(0, 0), (1, 1)]
    assert plan.obs_rows[0] == [1, 2 | (1 << 2), 0]  # [n_obs, word(obs_0), padding]
    assert [unpack_obs_codes([w], 2) for w in plan.obs_rows[1][1:]] == [list(c) for c in plan.obs_codes[1]]

    code = generate_sync_hook([plan])
    _assert_no_provider_imports(code)
    job = _StubJob()
    _exec_hook(code, job)

    # Per binding: one parameter-row push, then one observable-row push (no per-qubit pushes).
    assert job.streamed == [
        ("param_table_0_row", [0.1]),
        ("obs_row_0", plan.obs_rows[0]),
        ("param_table_0_row", [0.2]),
        ("obs_row_0", plan.obs_rows[1]),
    ]


def test_estimator_observables_only_shared_row():
    plan = _estimator_plan([SparsePauliOp(["ZX", "XZ"])])
    assert plan.shared_obs_row and plan.streams_inputs and not plan.streams_params

    code = generate_sync_hook([plan])
    _assert_no_provider_imports(code)
    job = _StubJob()
    _exec_hook(code, job)

    assert job.streamed == [("obs_row_0", plan.obs_rows[0])]
    assert plan.obs_rows[0][0] == 2


def test_estimator_io_obs_row_sends_only_used_prefix():
    plan = _estimator_plan([SparsePauliOp("XY"), SparsePauliOp(["ZZ", "XX"])], values=[[0.1], [0.2]], input_type=InputType.IO1)
    code = generate_sync_hook([plan])
    job = _StubJob()
    _exec_hook(code, job)

    expected = [0.1, *plan.obs_rows[0][:2], 0.2, *plan.obs_rows[1][:3]]  # binding 0 drops its padding word
    assert job.io_values == [{"io1": v} for v in expected]
    assert job.resumed == len(expected)
