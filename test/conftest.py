"""Shared pytest fixtures for qiskit_qm_provider tests."""

import pathlib
import pytest

from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
    FluxTunableQuam,
)

# Canonical QuAM state fixture — copied from the real QPU config.
QUAM_STATE_DIR = pathlib.Path(__file__).parent / "fixtures" / "quam_state"


@pytest.fixture(scope="session")
def quam_machine():
    """Load the pinned QuAM machine from the in-tree fixture folder."""
    return FluxTunableQuam.load(QUAM_STATE_DIR)


@pytest.fixture(scope="session")
def flux_tunable_backend(quam_machine):
    """Create a backend via QMProvider and seed basic macros from the QuAM state."""
    from qiskit_qm_provider import QMProvider
    from qiskit_qm_provider.backend.flux_tunable_transmon_backend import (
        FluxTunableTransmonBackend,
    )
    from qiskit_qm_provider.backend.backend_utils import add_basic_macros

    provider = QMProvider()
    backend = provider.get_backend(quam_machine, FluxTunableTransmonBackend)
    add_basic_macros(backend)
    return backend


@pytest.fixture(autouse=True)
def _reset_max_circuits(flux_tunable_backend):
    """Restore max_circuits=30 after every test that mutates it."""
    yield
    flux_tunable_backend.set_options(max_circuits=30)


@pytest.fixture(scope="session")
def qm_provider():
    """Return a bare QMProvider."""
    from qiskit_qm_provider import QMProvider

    return QMProvider()
