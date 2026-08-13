"""Tests for unified QM program submission across local, cloud, and OPX queues."""

from unittest.mock import MagicMock, call

import pytest

from qm import QuantumMachinesManager, SimulationConfig

from qiskit_qm_provider.job.qm_execution_options import (
    cloud_execute_options,
    enqueue_program,
    ensure_job_running,
    execute_kwargs_for_qmm,
    is_cloud_quantum_machines_manager,
    job_status_string,
    queue_kwargs_for_qmm,
    should_execute_programs,
    submit_qua_programs,
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

    def test_execute_kwargs_local_with_simulate(self, local_qmm):
        simulate = SimulationConfig(duration=100)
        metadata = {"compiler_options": {"foo": "bar"}, "simulate": simulate, "timeout": 60}
        assert execute_kwargs_for_qmm(local_qmm, metadata) == {
            "compiler_options": {"foo": "bar"},
            "simulate": simulate,
        }

    def test_execute_kwargs_local_without_simulate(self, local_qmm):
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

    def test_queue_kwargs_omit_none_compiler_options(self, local_qmm, cloud_qmm):
        assert queue_kwargs_for_qmm(local_qmm, {"compiler_options": None}) == {}
        assert queue_kwargs_for_qmm(cloud_qmm, {"compiler_options": {"x": 1}}) == {}
        assert queue_kwargs_for_qmm(local_qmm, {"compiler_options": {"x": 1}}) == {
            "compiler_options": {"x": 1},
        }

    def test_cloud_execute_options_timeout_only(self):
        assert cloud_execute_options({"timeout": 30}) == {"timeout": 30}
        assert cloud_execute_options({}) == {}

    def test_should_execute_programs(self, cloud_qmm, local_qmm):
        assert should_execute_programs(cloud_qmm, {})
        assert should_execute_programs(local_qmm, {"simulate": SimulationConfig(duration=1)})
        assert not should_execute_programs(local_qmm, {"simulate": None})


class TestEnqueueAndSubmit:
    def test_enqueue_prefers_add_to_queue(self):
        qm = MagicMock()
        qm.add_to_queue.return_value = "opx1000-job"
        program = MagicMock()
        assert enqueue_program(qm, program, compiler_options={"a": 1}) == "opx1000-job"
        qm.add_to_queue.assert_called_once_with(program, compiler_options={"a": 1})
        qm.queue.add.assert_not_called()

    def test_enqueue_falls_back_to_queue_add(self):
        qm = MagicMock(spec=["queue"])
        qm.queue.add.return_value = "opx-plus-job"
        program = MagicMock()
        assert enqueue_program(qm, program, compiler_options={"a": 1}) == "opx-plus-job"
        qm.queue.add.assert_called_once_with(program, compiler_options={"a": 1})

    def test_submit_cloud_uses_execute(self, cloud_qmm):
        qm = MagicMock()
        qm.execute.return_value = MagicMock(id="c1")
        programs = [MagicMock(), MagicMock()]
        jobs = submit_qua_programs(
            qm, cloud_qmm, programs, {"compiler_options": {"x": 1}, "timeout": 9}
        )
        assert len(jobs) == 2
        assert qm.execute.call_args_list == [
            call(programs[0], options={"timeout": 9}),
            call(programs[1], options={"timeout": 9}),
        ]

    def test_submit_simulate_uses_execute(self, local_qmm):
        qm = MagicMock()
        simulate = SimulationConfig(duration=10)
        program = MagicMock()
        submit_qua_programs(
            qm,
            local_qmm,
            [program],
            {"simulate": simulate, "compiler_options": None},
        )
        qm.execute.assert_called_once_with(
            program, compiler_options=None, simulate=simulate
        )
        qm.add_to_queue.assert_not_called()
        qm.queue.add.assert_not_called()

    def test_submit_real_hardware_queues(self, local_qmm):
        qm = MagicMock()
        qm.add_to_queue.side_effect = lambda prog, **kw: MagicMock(id=f"q-{id(prog)}")
        programs = [MagicMock(), MagicMock()]
        jobs = submit_qua_programs(
            qm, local_qmm, programs, {"simulate": None, "compiler_options": {"c": 1}}
        )
        assert len(jobs) == 2
        assert qm.add_to_queue.call_count == 2
        qm.execute.assert_not_called()

    def test_submit_cloud_rejects_simulate(self, cloud_qmm):
        with pytest.raises(ValueError, match="SimulationConfig"):
            submit_qua_programs(
                MagicMock(),
                cloud_qmm,
                [MagicMock()],
                {"simulate": SimulationConfig(duration=1)},
            )


class TestEnsureJobRunningAndStatus:
    def test_ensure_job_running_pending(self):
        pending = MagicMock()
        running = MagicMock()
        pending.wait_for_execution.return_value = running
        assert ensure_job_running(pending) is running

    def test_ensure_job_running_job_api(self):
        job = MagicMock(spec=["wait_until", "id"])
        assert ensure_job_running(job) is job
        job.wait_until.assert_called_once_with("Running")

    def test_job_status_string_from_get_status(self):
        job = MagicMock(spec=["get_status"])
        job.get_status.return_value = "In queue"
        assert job_status_string(job) == "in queue"


class TestQMJobSubmit:
    def test_submit_cloud_execute_without_compiler_options(self):
        from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager
        from qiskit_qm_provider.job.qm_job import QMJob

        cloud_qm = MagicMock()
        cloud_qm.execute.return_value = MagicMock(id="cloud-job-1")

        class FakeBackend:
            qmm = object.__new__(CloudQuantumMachinesManager)

        job = QMJob(
            FakeBackend(),
            "pending",
            cloud_qm,
            MagicMock(),
            result_function=MagicMock(),
            compiler_options={"should_not_pass": True},
            timeout=90,
        )
        job.submit()
        cloud_qm.execute.assert_called_once_with(
            job.programs[0], options={"timeout": 90}
        )

    def test_submit_real_hardware_uses_queue(self, local_qmm):
        from qiskit_qm_provider.job.qm_job import QMJob

        qm = MagicMock()
        qm.add_to_queue.return_value = MagicMock(id="queued-1")

        class FakeBackend:
            qmm = local_qmm

        program = MagicMock()
        job = QMJob(
            FakeBackend(),
            "pending",
            qm,
            program,
            result_function=MagicMock(),
            compiler_options={"opt": True},
            simulate=None,
        )
        job.submit()
        qm.add_to_queue.assert_called_once_with(program, compiler_options={"opt": True})
        qm.execute.assert_not_called()
