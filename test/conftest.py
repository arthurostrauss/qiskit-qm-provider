"""Shared pytest fixtures for qiskit_qm_provider tests.

Run the suite with the same interpreter that has qiskit, QM, and provider deps installed
(e.g. ``~/Documents/.venv/bin/python -m pytest``) to avoid collection/import failures from
a mismatched default ``python``.
"""

import pytest

try:
    from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
        FluxTunableQuam,
    )
except ImportError:  # optional dev dependency
    FluxTunableQuam = None  # type: ignore[misc, assignment]


@pytest.fixture(scope="session")
def quam_machine():
    """Load a QuAM machine directly (no state folder path)."""
    if FluxTunableQuam is None:
        pytest.skip("quam_builder is not installed")
    return FluxTunableQuam.load()


@pytest.fixture(scope="session")
def flux_tunable_backend(quam_machine):
    """Create a FluxTunableTransmonBackend from the loaded QuAM machine."""
    from qiskit_qm_provider.backend.flux_tunable_transmon_backend import (
        FluxTunableTransmonBackend,
    )

    return FluxTunableTransmonBackend(quam_machine)


@pytest.fixture(scope="session")
def qm_provider():
    """Create a QMProvider without a state folder path."""
    from qiskit_qm_provider.providers.qm_provider import QMProvider

    return QMProvider()
