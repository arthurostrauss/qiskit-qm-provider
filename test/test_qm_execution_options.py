"""Tests for QM execution option trimming across local and cloud QMM backends."""

from unittest.mock import MagicMock

import pytest

from qm import QuantumMachinesManager

from qiskit_qm_provider.job.qm_execution_options import (
    cloud_execute_options,
    compile_kwargs_for_qmm,
    execute_kwargs_for_qmm,
    is_cloud_quantum_machines_manager,
    simulate_kwargs_for_qmm,
    trimmed_compiler_options,
)


@pytest.fixture
def cloud_qmm():
    from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager

    return object.__new__(CloudQuantumMachinesManager)


@pytest.fixture
def local_qmm():
    return MagicMock(spec=QuantumMachinesManager)


class TestQmExecutionOptions:
    def test_is_cloud_quantum_machines_manager(self, cloud_qmm, local_qmm):
        assert is_cloud_quantum_machines_manager(cloud_qmm)
        assert not is_cloud_quantum_machines_manager(local_qmm)

    def test_trimmed_compiler_options_local(self, local_qmm):
        metadata = {"compiler_options": {"foo": "bar"}}
        assert trimmed_compiler_options(local_qmm, metadata) == {"foo": "bar"}

    def test_trimmed_compiler_options_cloud(self, cloud_qmm):
        metadata = {"compiler_options": {"foo": "bar"}}
        assert trimmed_compiler_options(cloud_qmm, metadata) is None

    def test_execute_kwargs_local_passes_compiler_options(self, local_qmm):
        metadata = {"compiler_options": None, "timeout": 60}
        assert execute_kwargs_for_qmm(local_qmm, metadata) == {
            "compiler_options": None,
        }

    def test_execute_kwargs_cloud_uses_options_dict(self, cloud_qmm):
        metadata = {"compiler_options": {"foo": "bar"}, "timeout": 120}
        assert execute_kwargs_for_qmm(cloud_qmm, metadata) == {
            "options": {"timeout": 120},
        }

    def test_execute_kwargs_cloud_omits_empty_options(self, cloud_qmm):
        metadata = {"compiler_options": {"foo": "bar"}}
        assert execute_kwargs_for_qmm(cloud_qmm, metadata) == {}

    def test_compile_kwargs_cloud_empty(self, cloud_qmm):
        metadata = {"compiler_options": {"foo": "bar"}}
        assert compile_kwargs_for_qmm(cloud_qmm, metadata) == {}

    def test_simulate_kwargs_match_compile_trim(self, cloud_qmm, local_qmm):
        metadata = {"compiler_options": {"foo": "bar"}}
        assert simulate_kwargs_for_qmm(cloud_qmm, metadata) == {}
        assert simulate_kwargs_for_qmm(local_qmm, metadata) == {
            "compiler_options": {"foo": "bar"},
        }

    def test_cloud_execute_options_timeout_only(self):
        assert cloud_execute_options({"timeout": 30}) == {"timeout": 30}
        assert cloud_execute_options({}) == {}


class TestQMJobSubmitCloudExecute:
    def test_submit_cloud_execute_without_compiler_options(self):
        from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager
        from qiskit_qm_provider.job.qm_job import QMJob

        cloud_qm = MagicMock()
        cloud_qm.execute.return_value = MagicMock(id="cloud-job-1")

        class FakeBackend:
            qmm = object.__new__(CloudQuantumMachinesManager)

        backend = FakeBackend()
        program = MagicMock()

        job = QMJob(
            backend,
            "pending",
            cloud_qm,
            program,
            result_function=MagicMock(),
            compiler_options={"should_not_pass": True},
            timeout=90,
        )
        job.submit()

        cloud_qm.execute.assert_called_once_with(program, options={"timeout": 90})
        assert job._qm_jobs is not None
