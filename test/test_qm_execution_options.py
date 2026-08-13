"""Tests for unified QM program submission across local, cloud, and OPX queues."""

from unittest.mock import MagicMock, call

import pytest

from qm import QuantumMachinesManager, SimulationConfig

from qiskit_qm_provider.job.iqcc_job_mixin import aggregate_job_statuses
from qiskit_qm_provider.job.qm_execution_options import (
    await_running_jobs,
    enqueue_program,
    ensure_job_running,
    is_cloud_quantum_machines_manager,
    join_job_ids,
    submit_qua_programs,
)


@pytest.fixture
def cloud_qmm():
    from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager

    return object.__new__(CloudQuantumMachinesManager)


@pytest.fixture
def local_qmm():
    return MagicMock(spec=QuantumMachinesManager)


class TestSubmitQuaPrograms:
    def test_cloud_uses_execute_without_compiler_options(self, cloud_qmm):
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

    def test_cloud_omits_empty_options(self, cloud_qmm):
        qm = MagicMock()
        program = MagicMock()
        submit_qua_programs(qm, cloud_qmm, [program], {"compiler_options": {"x": 1}})
        qm.execute.assert_called_once_with(program)

    def test_simulate_uses_execute(self, local_qmm):
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

    def test_real_hardware_queues_with_compiler_options(self, local_qmm):
        qm = MagicMock()
        programs = [MagicMock(), MagicMock()]
        jobs = submit_qua_programs(
            qm, local_qmm, programs, {"simulate": None, "compiler_options": {"c": 1}}
        )
        assert len(jobs) == 2
        assert qm.add_to_queue.call_count == 2
        qm.add_to_queue.assert_any_call(programs[0], compiler_options={"c": 1})
        qm.execute.assert_not_called()

    def test_real_hardware_omits_none_compiler_options(self, local_qmm):
        qm = MagicMock()
        program = MagicMock()
        submit_qua_programs(
            qm, local_qmm, [program], {"simulate": None, "compiler_options": None}
        )
        qm.add_to_queue.assert_called_once_with(program)

    def test_cloud_rejects_simulate(self, cloud_qmm):
        with pytest.raises(ValueError, match="SimulationConfig"):
            submit_qua_programs(
                MagicMock(),
                cloud_qmm,
                [MagicMock()],
                {"simulate": SimulationConfig(duration=1)},
            )

    def test_is_cloud_detection(self, cloud_qmm, local_qmm):
        assert is_cloud_quantum_machines_manager(cloud_qmm)
        assert not is_cloud_quantum_machines_manager(local_qmm)


class TestEnqueueAndAwait:
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

    def test_ensure_job_running_pending(self):
        pending = MagicMock()
        running = MagicMock()
        pending.wait_for_execution.return_value = running
        assert ensure_job_running(pending) is running

    def test_ensure_job_running_job_api(self):
        job = MagicMock(spec=["wait_until", "id"])
        assert ensure_job_running(job) is job
        job.wait_until.assert_called_once_with("Running")

    def test_await_running_jobs_wraps_chunk_errors(self):
        pending = MagicMock()
        pending.wait_for_execution.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="Chunk 0 of 1 \\(PUB indices \\[3\\]\\)"):
            await_running_jobs([pending], [[3]], entity="PUB indices")

    def test_join_job_ids(self):
        assert join_job_ids([MagicMock(id="a"), MagicMock(id="b")]) == "a,b"
        assert join_job_ids([MagicMock(spec=[]), MagicMock(id="b")]) == "b"


class TestStatusAggregation:
    def test_opx1000_status_strings(self):
        from qiskit.providers import JobStatus

        job = MagicMock(spec=["get_status"])
        job.get_status.return_value = "In queue"
        assert aggregate_job_statuses([job]) == JobStatus.QUEUED

        job.get_status.return_value = "Done"
        assert aggregate_job_statuses([job]) == JobStatus.DONE


class TestQMJobSubmit:
    def test_submit_cloud_execute_without_compiler_options(self):
        from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager
        from qiskit_qm_provider.job.qm_job import CloudQMJob, QMJob

        cloud_qm = MagicMock()
        cloud_qm.execute.return_value = MagicMock(id="cloud-job-1")

        class FakeBackend:
            qmm = object.__new__(CloudQuantumMachinesManager)

        job = CloudQMJob(
            FakeBackend(),
            "pending",
            cloud_qm,
            MagicMock(),
            result_function=MagicMock(),
            compiler_options={"should_not_pass": True},
            timeout=90,
        )
        assert isinstance(job, QMJob)
        job.submit()
        cloud_qm.execute.assert_called_once_with(
            job.programs[0], options={"timeout": 90}
        )
        assert job.job_id() == "cloud-job-1"

    def test_from_circuits_selects_cloud_qm_job(self):
        from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager
        from qiskit_qm_provider.job.qm_job import CloudQMJob, QMJob
        from qiskit_qm_provider.job.qm_execution_options import is_cloud_quantum_machines_manager

        assert is_cloud_quantum_machines_manager(object.__new__(CloudQuantumMachinesManager))
        assert issubclass(CloudQMJob, QMJob)

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
