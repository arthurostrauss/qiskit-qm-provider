"""Shared pytest fixtures for qiskit_qm_provider tests.

Run the suite with the same interpreter that has qiskit, QM, and provider deps installed
(e.g. ``~/Documents/.venv/bin/python -m pytest``) to avoid collection/import failures from
a mismatched default ``python``.
"""

import os

import pytest

_QUAM_STATE_PATH = os.environ.get("QUAM_STATE_PATH")


@pytest.fixture(autouse=True)
def _reset_parameter_pool():
    """Isolate the process-global ``ParameterPool`` (registry, bound Quarc module, and
    Quarc's stream-id counters) around every test. Without this, name/struct state and
    monotonic stream ids leak across tests and make outcomes order-dependent."""
    from qiskit_qm_provider.parameter_table.parameter_pool import ParameterPool

    ParameterPool.reset()
    yield
    ParameterPool.reset()


try:
    from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
        FluxTunableQuam,
    )
except ImportError:  # optional dev dependency
    FluxTunableQuam = None  # type: ignore[misc, assignment]


@pytest.fixture(scope="session")
def quam_machine():
    """Load a QuAM machine from QUAM_STATE_PATH."""
    if FluxTunableQuam is None:
        pytest.skip("quam_builder is not installed")
    if not _QUAM_STATE_PATH:
        pytest.skip("QUAM_STATE_PATH env var not set — export it to the quam state directory")
    return FluxTunableQuam.load(_QUAM_STATE_PATH)


@pytest.fixture(scope="session")
def flux_tunable_backend(quam_machine):
    """Create a FluxTunableTransmonBackend from the loaded QuAM machine."""
    from qiskit_qm_provider.backend.flux_tunable_transmon_backend import (
        FluxTunableTransmonBackend,
    )

    return FluxTunableTransmonBackend(quam_machine)


@pytest.fixture(scope="session")
def qm_provider():
    """Create a QMProvider from QUAM_STATE_PATH."""
    if not _QUAM_STATE_PATH:
        pytest.skip("QUAM_STATE_PATH env var not set — export it to the quam state directory")
    from qiskit_qm_provider.providers.qm_provider import QMProvider

    return QMProvider(state_folder_path=_QUAM_STATE_PATH)
