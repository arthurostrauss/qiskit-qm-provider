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
def _reset_max_circuits(request):
    """Restore max_circuits=30 after every test that mutates it.

    Only resets when the test already depends on flux_tunable_backend, so tests
    that do not use the backend do not trigger a skip when QUAM_STATE_PATH is unset.
    """
    yield
    if _QUAM_STATE_PATH and "flux_tunable_backend" in request.fixturenames:
        backend = request.getfixturevalue("flux_tunable_backend")
        backend.set_options(max_circuits=30)


@pytest.fixture(scope="session")
def qm_provider():
    """Create a QMProvider from QUAM_STATE_PATH."""
    if not _QUAM_STATE_PATH:
        pytest.skip("QUAM_STATE_PATH env var not set — export it to the quam state directory")
    from qiskit_qm_provider.providers.qm_provider import QMProvider

    return QMProvider(state_folder_path=_QUAM_STATE_PATH)
