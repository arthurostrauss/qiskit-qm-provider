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

IQCC sync-hook submission is **not** handled here — see ``IQCCSamplerJob`` /
``IQCCEstimatorJob`` / ``IQCCJob``.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from qm import Program, SimulationConfig, QuantumMachinesManager

try:
    from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager  # type: ignore[import]
except ImportError:
    CloudQuantumMachinesManager = None  # type: ignore[misc, assignment]


def is_cloud_quantum_machines_manager(qmm: Any) -> bool:
    """Return whether *qmm* is an IQCC cloud :class:`CloudQuantumMachinesManager`."""
    return CloudQuantumMachinesManager is not None and isinstance(
        qmm, CloudQuantumMachinesManager
    )


def _execute_kwargs(qmm: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Kwargs for ``qm.execute``, trimmed for local vs cloud QMM."""
    if is_cloud_quantum_machines_manager(qmm):
        timeout = metadata.get("timeout")
        return {"options": {"timeout": timeout}} if timeout is not None else {}

    kwargs: dict[str, Any] = {
        "compiler_options": metadata.get("compiler_options", None),
    }
    simulate = metadata.get("simulate", None)
    if isinstance(simulate, SimulationConfig):
        kwargs["simulate"] = simulate
    return kwargs


def enqueue_program(qm: Any, program: Program, **queue_kwargs: Any) -> Any:
    """Add *program* to the OPX queue (OPX1000 ``add_to_queue``, else OPX+ ``queue.add``)."""
    try:
        return qm.add_to_queue(program, **queue_kwargs)
    except AttributeError:
        return qm.queue.add(program, **queue_kwargs)


def submit_qua_programs(
    qm: Any,
    qmm: QuantumMachinesManager | Any,
    programs: Sequence[Program],
    metadata: Mapping[str, Any],
) -> List[Any]:
    """Submit QUA *programs* on *qm*, returning SDK job handles.

    * Cloud or ``SimulationConfig`` → ``qm.execute`` with trimmed kwargs.
    * Real QuantumMachine → ``add_to_queue`` / ``queue.add`` for every program.
    """
    simulate = metadata.get("simulate", None)
    cloud = is_cloud_quantum_machines_manager(qmm)

    if cloud and isinstance(simulate, SimulationConfig):
        raise ValueError(
            "SimulationConfig is not supported for CloudQuantumMachinesManager backends"
        )

    if cloud or isinstance(simulate, SimulationConfig):
        execute_kwargs = _execute_kwargs(qmm, metadata)
        return [qm.execute(prog, **execute_kwargs) for prog in programs]

    queue_kwargs: dict[str, Any] = {}
    compiler_options = metadata.get("compiler_options", None)
    if compiler_options is not None:
        queue_kwargs["compiler_options"] = compiler_options
    return [enqueue_program(qm, prog, **queue_kwargs) for prog in programs]


def ensure_job_running(job: Any) -> Any:
    """Block until *job* has started; return a running handle.

    Supports OPX+ ``wait_for_execution`` and OPX1000 ``wait_until("Running")``.
    Already-running jobs (e.g. from ``execute``) are returned unchanged.
    """
    wait_for_execution = getattr(job, "wait_for_execution", None)
    if callable(wait_for_execution):
        return wait_for_execution()

    wait_until = getattr(job, "wait_until", None)
    if callable(wait_until):
        wait_until("Running")
        return job

    return job


def join_job_ids(jobs: Sequence[Any]) -> str:
    """Comma-join SDK job ids, stripping empty trailing entries."""
    return ",".join(str(getattr(j, "id", "") or "") for j in jobs).strip(",")


def await_running_jobs(
    pending_jobs: Sequence[Any],
    chunk_layout: Sequence[Sequence[int]],
    *,
    entity: str,
) -> List[Any]:
    """Wait until each pending chunk job is running; raise on per-chunk failure."""
    running_jobs: List[Any] = []
    for i, (pending, chunk) in enumerate(zip(pending_jobs, chunk_layout)):
        try:
            running_jobs.append(ensure_job_running(pending))
        except Exception as exc:
            raise RuntimeError(
                f"Chunk {i} of {len(pending_jobs)} ({entity} {chunk}) "
                f"failed to start execution"
            ) from exc
    return running_jobs
