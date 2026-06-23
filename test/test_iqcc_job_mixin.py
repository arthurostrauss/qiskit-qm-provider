"""Tests for IQCC cloud job diagnostics."""

import pytest

from qiskit_qm_provider.job.iqcc_job_mixin import (
    IQCCCloudExecutionError,
    IQCCJobMixin,
    iqcc_run_data_from_qm_job,
    raise_if_iqcc_cloud_failed,
)


SAMPLE_STDERR = """Traceback (most recent call last):
  File "/tmp/script.py", line 41, in <module>
    qm = qmm.open_qm(config, close_other_machines=True)
qm.exceptions.OpenQmException: Can not open QM
"""

SAMPLE_RUN_DATA = {
    "stdout": '2026-06-22 - qm - ERROR - PHYSICAL CONFIG ERROR in key "controllers.fems"\n',
    "stderr": SAMPLE_STDERR,
    "result": {"__qpu_execution_time_seconds": 0},
}


class _FakeCloudJob:
    def __init__(self, run_data):
        self._run_data = run_data


class _FakeIQCCJob(IQCCJobMixin):
    def __init__(self, qm_job):
        self._qm_jobs = [qm_job] if qm_job is not None else None

    def _parent_result(self):
        return "ok"

    def result(self):
        if self._qm_jobs is None:
            raise RuntimeError("IQCC job has not been submitted yet")
        self._check_iqcc_cloud_execution()
        return self._parent_result()


class TestIqccRunDataHelpers:
    def test_run_data_from_qm_job(self):
        job = _FakeCloudJob(SAMPLE_RUN_DATA)
        assert iqcc_run_data_from_qm_job(job) == SAMPLE_RUN_DATA

    def test_run_data_missing_when_not_dict(self):
        job = _FakeCloudJob(run_data="not-a-dict")
        assert iqcc_run_data_from_qm_job(job) is None

    def test_raise_if_traceback_in_stderr(self):
        with pytest.raises(IQCCCloudExecutionError) as exc_info:
            raise_if_iqcc_cloud_failed(_FakeCloudJob(SAMPLE_RUN_DATA))
        assert exc_info.value.args[0] == SAMPLE_STDERR.strip()
        assert exc_info.value.run_data == SAMPLE_RUN_DATA

    def test_no_raise_without_traceback(self):
        run_data = {"stderr": "warning only", "stdout": ""}
        raise_if_iqcc_cloud_failed(_FakeCloudJob(run_data))

    def test_no_raise_when_run_data_missing(self):
        raise_if_iqcc_cloud_failed(_FakeCloudJob(None))
        raise_if_iqcc_cloud_failed(object())


class TestIQCCJobMixin:
    def test_run_data_property(self):
        qm_job = _FakeCloudJob(SAMPLE_RUN_DATA)
        wrapper = _FakeIQCCJob(qm_job)
        assert wrapper.run_data == SAMPLE_RUN_DATA

    def test_run_data_none_before_submit(self):
        wrapper = _FakeIQCCJob(None)
        assert wrapper.run_data is None

    def test_result_raises_cloud_error_before_parent(self):
        wrapper = _FakeIQCCJob(_FakeCloudJob(SAMPLE_RUN_DATA))
        with pytest.raises(IQCCCloudExecutionError, match="OpenQmException"):
            wrapper.result()

    def test_result_delegates_when_cloud_ok(self):
        run_data = {"stderr": "", "stdout": "ok\n", "result": {}}
        wrapper = _FakeIQCCJob(_FakeCloudJob(run_data))
        assert wrapper.result() == "ok"
