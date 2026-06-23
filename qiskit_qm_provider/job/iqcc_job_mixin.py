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

"""Shared IQCC cloud job diagnostics and error surfacing."""

from __future__ import annotations

from typing import Any


def result_handles_from_qm_job(qm_job: Any) -> Any:
    """Return ``qm_job.result_handles``, or a per-job list for multi-program execution.

    Raises:
        RuntimeError: If ``qm_job`` is ``None`` (job not submitted yet).
    """
    if qm_job is None:
        raise RuntimeError("QM job has not submitted yet")
    if isinstance(qm_job, list):
        return [job.result_handles for job in qm_job]
    return qm_job.result_handles


class IQCCCloudExecutionError(RuntimeError):
    """IQCC cloud execution failed before measurement streams were produced.

    The exception message is the cloud ``stderr`` payload (typically a Python
    traceback from the remote runtime). Inspect :attr:`run_data` for the full
    IQCC record, including ``stdout`` and timing metadata.
    """

    def __init__(self, stderr: str, run_data: dict[str, Any] | None = None):
        super().__init__(stderr.strip())
        self.stderr = stderr.strip()
        self.run_data = run_data


def iqcc_run_data_from_qm_job(qm_job: Any) -> dict[str, Any] | None:
    """Return ``qm_job._run_data`` when the IQCC cloud client populated it."""
    if qm_job is None:
        return None
    run_data = getattr(qm_job, "_run_data", None)
    return run_data if isinstance(run_data, dict) else None


def raise_if_iqcc_cloud_failed(qm_job: Any) -> None:
    """Raise :class:`IQCCCloudExecutionError` when cloud ``stderr`` contains a traceback."""
    run_data = iqcc_run_data_from_qm_job(qm_job)
    if run_data is None:
        return
    stderr = run_data.get("stderr")
    if isinstance(stderr, str) and "Traceback" in stderr:
        raise IQCCCloudExecutionError(stderr, run_data=run_data)


class IQCCJobMixin:
    """Mixin for IQCC wrapper jobs (:class:`IQCCJob`, primitive IQCC variants).

    Exposes the raw IQCC cloud execution record and re-raises remote failures
    instead of surfacing misleading local ``KeyError``s on missing stream keys.
    """

    _qm_job: Any

    @property
    def run_data(self) -> dict[str, Any] | None:
        """Raw IQCC cloud execution record from ``qm_job._run_data``.

        When present, keys typically include ``stdout``, ``stderr``, and
        ``result`` (timing / fridge metadata). ``None`` if the job has not been
        submitted yet or the cloud client has not attached run data.
        """
        return iqcc_run_data_from_qm_job(getattr(self, "_qm_job", None))

    def _check_iqcc_cloud_execution(self) -> None:
        raise_if_iqcc_cloud_failed(getattr(self, "_qm_job", None))

    def result(self):
        if getattr(self, "_qm_job", None) is None:
            raise RuntimeError("IQCC job has not been submitted yet")
        self._check_iqcc_cloud_execution()
        return super().result()
