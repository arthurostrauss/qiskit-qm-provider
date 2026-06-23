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
``qiskit_qm_provider`` or ``numpy``. These tests build small parameter tables and
fake pubs/execution plans, render the hooks, and exec them against a stub job to
verify the right QM-job calls are made with the right coercions.
"""

import numpy as np
import pytest
from qm.qua import fixed

from qiskit_qm_provider.parameter_table import InputType, ParameterTable
from qiskit_qm_provider.job.post_hook_sampler import generate_sync_hook_sampler
from qiskit_qm_provider.job.post_hook_estimator import generate_sync_hook_estimator


class _FakeArray:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def ravel(self):
        return self

    def as_array(self):
        return self._arr


class _FakePub:
    def __init__(self, values):
        self.parameter_values = _FakeArray(values)


class _FakePlan:
    def __init__(self, pub, param_table, observables_var, obs_indices):
        self.pub = pub
        self.param_table = param_table
        self.observables_var = observables_var
        self.obs_indices = obs_indices


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
    pub = _FakePub([[0.1, 2.0], [0.3, 4.0]])
    code = generate_sync_hook_sampler([pub], [table])
    _assert_no_provider_imports(code)

    job = _StubJob()
    _exec_hook(code, job)

    # Two parameter sets, two params each -> 4 stream pushes, with int/fixed coercion.
    assert job.streamed == [("x", 0.1), ("n", 2), ("x", 0.3), ("n", 4)]
    assert all(isinstance(v, float) for (n, v) in job.streamed if n == "x")
    assert all(isinstance(v, int) for (n, v) in job.streamed if n == "n")


def test_sampler_io_pushes_pause_resume():
    table = ParameterTable({"a": (0.0, fixed, InputType.IO1)}, name="io")
    pub = _FakePub([[0.5]])
    code = generate_sync_hook_sampler([pub], [table])
    _assert_no_provider_imports(code)

    job = _StubJob()
    _exec_hook(code, job)

    assert job.io_values == [{"io1": 0.5}]
    assert job.resumed == 1


def test_sampler_none_table_pushes_nothing():
    pub = _FakePub([[0.1]])
    code = generate_sync_hook_sampler([pub], [None])
    _assert_no_provider_imports(code)

    job = _StubJob()
    _exec_hook(code, job)
    assert job.streamed == []
    assert job.io_values == []


def test_estimator_pushes_params_and_observables():
    param_table = ParameterTable({"theta": (0.0, fixed, InputType.INPUT_STREAM)}, name="p")
    obs_var = ParameterTable(
        {"obs_0": (0, int, InputType.INPUT_STREAM), "obs_1": (0, int, InputType.INPUT_STREAM)},
        name="obs",
    )
    obs_length_var = ParameterTable({"obs_length_var": (0, int, InputType.INPUT_STREAM)}, name="L").get_parameter(
        "obs_length_var"
    )

    pub = _FakePub([[0.1], [0.2]])
    # One obs-index list per parameter set; each is a list of 2-qubit tuples.
    obs_indices = [[(1, 2)], [(3, 0), (1, 1)]]
    plan = _FakePlan(pub, param_table, obs_var, obs_indices)

    code = generate_sync_hook_estimator([plan], obs_length_var)
    _assert_no_provider_imports(code)

    job = _StubJob()
    _exec_hook(code, job)

    # param set 0: theta=0.1, obs_length=1, obs_0/obs_1 = (1,2)
    # param set 1: theta=0.2, obs_length=2, then (3,0) and (1,1)
    assert job.streamed == [
        ("theta", 0.1),
        ("obs_length_var", 1),
        ("obs_0", 1),
        ("obs_1", 2),
        ("theta", 0.2),
        ("obs_length_var", 2),
        ("obs_0", 3),
        ("obs_1", 0),
        ("obs_0", 1),
        ("obs_1", 1),
    ]


def test_estimator_observables_only_no_params():
    obs_var = ParameterTable(
        {"obs_0": (0, int, InputType.INPUT_STREAM), "obs_1": (0, int, InputType.INPUT_STREAM)},
        name="obs",
    )
    obs_length_var = ParameterTable({"obs_length_var": (0, int, InputType.INPUT_STREAM)}, name="L").get_parameter(
        "obs_length_var"
    )
    pub = _FakePub([])
    obs_indices = [[(0, 1), (2, 3)]]
    plan = _FakePlan(pub, None, obs_var, obs_indices)

    code = generate_sync_hook_estimator([plan], obs_length_var)
    _assert_no_provider_imports(code)

    job = _StubJob()
    _exec_hook(code, job)

    assert job.streamed == [
        ("obs_length_var", 2),
        ("obs_0", 0),
        ("obs_1", 1),
        ("obs_0", 2),
        ("obs_1", 3),
    ]
