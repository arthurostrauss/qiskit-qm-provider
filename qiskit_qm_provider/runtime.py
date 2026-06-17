"""Runtime helpers for hybrid classical/QUA workflows.

Provides job-access utilities that work across local QMBackend mode and
IQCC sync-hook mode without modifying quarc or the qm-qua SDK.
"""

from __future__ import annotations
import argparse
import logging
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from qm import QuantumMachinesManager


def _is_sync_hook_mode() -> bool:
    """True when the process was started by IQCC sync-hook (has -j/--jobId in argv)."""
    import sys

    return any(a in sys.argv for a in ("-j", "--jobId"))


def _job_from_sync_hook_args() -> Any:
    """Reconstruct a job from IQCC sync-hook CLI arguments (-j/-q/-i/-p)."""
    from qm import QuantumMachinesManager

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-j", "--jobId")
    parser.add_argument("-q", "--qmId")
    parser.add_argument("-i", "--ip")
    parser.add_argument("-p", "--port", type=int)
    args, _ = parser.parse_known_args()
    qmm = QuantumMachinesManager(host=args.ip, port=args.port, log_level=logging.ERROR)
    return qmm.get_job(args.jobId)


def _poll_running_job_from_qmm(qmm: QuantumMachinesManager, timeout: float, poll_interval: float) -> Any:
    """Poll the QMM for a running job until one appears or timeout expires.

    Two discovery strategies are tried in order:

    1. ``qmm.get_jobs(status=["Running"])`` — available on QOP 3.x; returns a
       lightweight :class:`~qm.api.v2.job_api.JobData` list from which we take the
       first entry and fetch the full :class:`~qm.api.v2.job_api.JobApi` via
       ``qmm.get_job(job_id)``.

    2. ``qmm.list_open_qms()`` fallback — for QOP 2.x where ``get_jobs`` raises
       :class:`NotImplementedError`; iterates all open machines and returns the first
       one that has a running job via ``qm.get_running_job()``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = _try_get_running_job(qmm)
        if job is not None:
            return job
        time.sleep(poll_interval)
    raise TimeoutError(f"No running job found on QMM within {timeout}s")


def _try_get_running_job(qmm: QuantumMachinesManager) -> Any:
    """Single probe attempt — returns the job or None."""
    # QOP 3.x path
    try:
        jobs = qmm.get_jobs(status=["Running"])
        if jobs:
            return qmm.get_job(jobs[0].id)
    except (NotImplementedError, Exception):
        pass

    # QOP 2.x / fallback path
    try:
        for qm_id in qmm.list_open_qms():
            qm = qmm.get_qm(machine_id=qm_id)
            job = qm.get_running_job()
            if job is not None:
                return job
    except Exception:
        pass

    return None
