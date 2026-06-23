"""Tests for job.result_handles delegation."""

import pytest

from qiskit_qm_provider.job.iqcc_job_mixin import result_handles_from_qm_job


class _FakeHandles:
    def __init__(self, name: str):
        self.name = name


class _FakeQmJob:
    def __init__(self, name: str = "main"):
        self.result_handles = _FakeHandles(name)


def test_result_handles_from_qm_job_single():
    job = _FakeQmJob("streams")
    assert result_handles_from_qm_job(job).name == "streams"


def test_result_handles_from_qm_job_list():
    jobs = [_FakeQmJob("a"), _FakeQmJob("b")]
    handles = result_handles_from_qm_job(jobs)
    assert [h.name for h in handles] == ["a", "b"]


def test_result_handles_from_qm_job_none_raises():
    with pytest.raises(RuntimeError, match="not submitted"):
        result_handles_from_qm_job(None)
