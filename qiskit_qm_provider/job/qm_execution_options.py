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

"""Unified QM program submission for local, cloud, and OPX+/OPX1000 queues.

Execution always goes through the quantum-machine instance:

* **Simulate / cloud** — ``qm.execute(...)`` with kwargs trimmed per QMM type.
  Local simulation uses ``execute(program, simulate=SimulationConfig, ...)``
  (equivalent to ``qm.simulate``); cloud uses ``execute(program, options=...)``.
* **Real QuantumMachine** (``simulate is None``) — queue the program(s). Prefer
  OPX1000 :meth:`~qm.api.v2.qm_api.QmApi.add_to_queue`; fall back to OPX+
  :meth:`~qm.jobs.job_queue_old_api.QmQueue.add`.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from qm import Program, SimulationConfig, QuantumMachinesManager

try:
    from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager  # type: ignore[import]
except ImportError:
    CloudQuantumMachinesManager = None  # type: ignore[misc, assignment]


def is_cloud_quantum_machines_manager(
    qmm: QuantumMachinesManager | CloudQuantumMachinesManager,
) -> bool:
    """Return whether *qmm* is an IQCC cloud :class:`CloudQuantumMachinesManager`."""
    return CloudQuantumMachinesManager is not None and isinstance(
        qmm, CloudQuantumMachinesManager
    )


def cloud_execute_options(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Build the ``options`` dict accepted by :meth:`CloudQuantumMachine.execute`."""
    options: dict[str, Any] = {}
    timeout = metadata.get("timeout")
    if timeout is not None:
        options["timeout"] = timeout
    return options


def execute_kwargs_for_qmm(
    qmm: QuantumMachinesManager | CloudQuantumMachinesManager,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Keyword arguments for ``qm.execute()`` trimmed to the active QMM type.

    Local / SaaS backends forward ``compiler_options`` and, when present, a
    ``SimulationConfig`` via the ``simulate`` keyword (same as ``qm.simulate``).
    Cloud backends only accept ``options={...}`` and never ``compiler_options``.
    """
    if is_cloud_quantum_machines_manager(qmm):
        cloud_opts = cloud_execute_options(metadata)
        return {"options": cloud_opts} if cloud_opts else {}

    kwargs: dict[str, Any] = {
        "compiler_options": metadata.get("compiler_options", None),
    }
    simulate = metadata.get("simulate", None)
    if isinstance(simulate, SimulationConfig):
        kwargs["simulate"] = simulate
    return kwargs


def queue_kwargs_for_qmm(
    qmm: QuantumMachinesManager | CloudQuantumMachinesManager,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Keyword arguments for ``add_to_queue`` / ``queue.add`` on real hardware."""
    if is_cloud_quantum_machines_manager(qmm):
        return {}
    compiler_options = metadata.get("compiler_options", None)
    if compiler_options is None:
        return {}
    return {"compiler_options": compiler_options}


def enqueue_program(qm: Any, program: Program, **queue_kwargs: Any) -> Any:
    """Add *program* to the OPX queue using OPX1000 API with OPX+ fallback.

    Tries :meth:`~qm.api.v2.qm_api.QmApi.add_to_queue` first. On
    :class:`AttributeError` (classic :class:`~qm.QuantumMachine` / OPX+), falls
    back to :meth:`~qm.jobs.job_queue_old_api.QmQueue.add`.
    """
    try:
        return qm.add_to_queue(program, **queue_kwargs)
    except AttributeError:
        return qm.queue.add(program, **queue_kwargs)


def should_execute_programs(
    qmm: QuantumMachinesManager | CloudQuantumMachinesManager,
    metadata: Mapping[str, Any],
) -> bool:
    """Whether programs should be submitted via ``qm.execute`` (vs. the queue).

    ``execute`` is used for cloud backends and for local simulation. Real
    hardware with ``simulate is None`` must use the queue — looping
    ``execute`` clears the queue between programs.
    """
    if is_cloud_quantum_machines_manager(qmm):
        return True
    return isinstance(metadata.get("simulate", None), SimulationConfig)


def submit_qua_programs(
    qm: Any,
    qmm: QuantumMachinesManager | CloudQuantumMachinesManager,
    programs: Sequence[Program],
    metadata: Mapping[str, Any],
) -> List[Any]:
    """Submit QUA *programs* on *qm*, returning the list of SDK job handles.

    * Cloud or ``SimulationConfig`` → ``qm.execute`` with trimmed kwargs.
    * Real QuantumMachine → ``add_to_queue`` / ``queue.add`` for every program.
    """
    simulate = metadata.get("simulate", None)
    if is_cloud_quantum_machines_manager(qmm) and isinstance(simulate, SimulationConfig):
        raise ValueError(
            "SimulationConfig is not supported for CloudQuantumMachinesManager backends"
        )

    if should_execute_programs(qmm, metadata):
        execute_kwargs = execute_kwargs_for_qmm(qmm, metadata)
        return [qm.execute(prog, **execute_kwargs) for prog in programs]

    queue_kwargs = queue_kwargs_for_qmm(qmm, metadata)
    return [enqueue_program(qm, prog, **queue_kwargs) for prog in programs]


def ensure_job_running(job: Any) -> Any:
    """Block until *job* has started execution; return a running job handle.

    Supports OPX+ :class:`~qm.jobs.pending_job.QmPendingJob`
    (``wait_for_execution``) and OPX1000 :class:`~qm.api.v2.job_api.job_api.JobApi`
    (``wait_until("Running")``). Already-running jobs are returned unchanged.
    """
    wait_for_execution = getattr(job, "wait_for_execution", None)
    if callable(wait_for_execution):
        return wait_for_execution()

    wait_until = getattr(job, "wait_until", None)
    if callable(wait_until):
        wait_until("Running")
        return job

    return job


def job_status_string(job: Any) -> str:
    """Normalize a QM SDK job status to a lowercase string for Qiskit mapping."""
    status = getattr(job, "status", None)
    if callable(status):
        status = status()
    elif status is None:
        get_status = getattr(job, "get_status", None)
        status = get_status() if callable(get_status) else "unknown"
    if status is None:
        return "unknown"
    return str(status).strip().lower()
