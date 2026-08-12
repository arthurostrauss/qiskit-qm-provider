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

"""Trim QM run metadata for local vs cloud QuantumMachinesManager execute APIs."""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

from qm import QuantumMachinesManager

if TYPE_CHECKING:
    from qm import CompilerOptionArguments

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


def trimmed_compiler_options(
    qmm: QuantumMachinesManager | CloudQuantumMachinesManager,
    metadata: Mapping[str, Any],
) -> CompilerOptionArguments | None:
    """Return ``compiler_options`` from *metadata* for local QMM backends, else ``None``.

    ``CloudQuantumMachine.compile`` / ``execute`` do not accept ``compiler_options``.
    """
    if is_cloud_quantum_machines_manager(qmm):
        return None
    return metadata.get("compiler_options", None)


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
    """Keyword arguments for ``qm.execute()`` trimmed to the active QMM type."""
    if is_cloud_quantum_machines_manager(qmm):
        cloud_opts = cloud_execute_options(metadata)
        return {"options": cloud_opts} if cloud_opts else {}
    compiler_options = metadata.get("compiler_options", None)
    return {"compiler_options": compiler_options}


def compile_kwargs_for_qmm(
    qmm: QuantumMachinesManager | CloudQuantumMachinesManager,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Keyword arguments for ``qm.compile()`` trimmed to the active QMM type."""
    compiler_options = trimmed_compiler_options(qmm, metadata)
    if compiler_options is None:
        return {}
    return {"compiler_options": compiler_options}


def simulate_kwargs_for_qmm(
    qmm: QuantumMachinesManager | CloudQuantumMachinesManager,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Keyword arguments for ``qm.simulate()`` / ``qmm.simulate()`` on local QMM backends."""
    compiler_options = trimmed_compiler_options(qmm, metadata)
    if compiler_options is None:
        return {}
    return {"compiler_options": compiler_options}
