"""Backends over a full dummy QuAM (see :mod:`test.dummy_quam`), with no QMM, state folder or hardware.

Importing this module skips the importing test module when quam-builder (or its builder
helpers, which need a version newer than v0.4.0) is unavailable.
"""

import pytest

pytest.importorskip("quam_builder.builder.superconducting.modify_quam")

from quam_builder.architecture.superconducting.qpu import FluxTunableQuam  # noqa: E402

from qiskit_qm_provider.backend.flux_tunable_transmon_backend import FluxTunableTransmonBackend  # noqa: E402

from .dummy_quam import build_dummy_quam_for_qubits  # noqa: E402


def build_machine(num_qubits: int = 2, quam_class=FluxTunableQuam):
    """A config-valid dummy QuAM with ``num_qubits`` qubits and nearest-neighbour pairs."""
    return build_dummy_quam_for_qubits(num_qubits, quam_class)


def build_backend(num_qubits: int = 2, conditional_plays=("x180", "-y90"), backend_class=FluxTunableTransmonBackend):
    """A backend over :func:`build_machine`, using the provider's ``add_basic_macros``
    (``x``, ``y``, ``sx``, ``sxdg``, ``sy``, ``sydg``, ``rz``, ``measure``, a thermal
    ``reset``, ``delay``, ``id``, and ``cz`` on every neighbouring pair).

    ``QUAM_STATE_PATH`` must point at a writable directory (use ``monkeypatch.setenv``).
    """
    backend = backend_class(build_machine(num_qubits))
    if conditional_plays:
        backend.register_conditional_play(*conditional_plays)
    return backend
