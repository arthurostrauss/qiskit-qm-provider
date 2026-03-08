"""Tests for QMProvider."""

import pytest
from qiskit_qm_provider.providers.qm_provider import QMProvider
from qiskit_qm_provider.backend.flux_tunable_transmon_backend import (
    FluxTunableTransmonBackend,
)
from qiskit_qm_provider.backend.qm_backend import QMBackend


class TestQMProviderInit:
    def test_default_init(self):
        provider = QMProvider()
        assert provider.state_folder_path is None
        assert provider._quam_cls is not None

    def test_init_with_path(self):
        provider = QMProvider(state_folder_path="/tmp/some_path")
        assert provider.state_folder_path == "/tmp/some_path"

    def test_init_with_custom_quam_cls(self):
        from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
            FluxTunableQuam,
        )

        provider = QMProvider(quam_cls=FluxTunableQuam)
        assert provider._quam_cls is FluxTunableQuam

    def test_default_quam_cls_is_flux_tunable(self):
        from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
            FluxTunableQuam,
        )

        provider = QMProvider()
        assert provider._quam_cls is FluxTunableQuam


class TestQMProviderGetMachine:
    def test_get_machine_returns_quam(self, qm_provider):
        machine = qm_provider.get_machine()
        assert machine is not None
        assert hasattr(machine, "qubits")
        assert hasattr(machine, "qubit_pairs")
        assert hasattr(machine, "active_qubits")
        assert hasattr(machine, "active_qubit_pairs")


class TestQMProviderGetBackend:
    def test_get_backend_default_is_base_qmbackend(self, qm_provider):
        backend = qm_provider.get_backend()
        assert type(backend) is QMBackend
        assert backend.machine is not None

    def test_get_backend_explicit_flux_tunable(self, qm_provider):
        backend = qm_provider.get_backend(backend_cls=FluxTunableTransmonBackend)
        assert isinstance(backend, FluxTunableTransmonBackend)
        assert isinstance(backend, QMBackend)

    def test_get_backend_invalid_machine_raises(self):
        provider = QMProvider()
        with pytest.raises(ValueError, match="Quam instance"):
            provider.get_backend(machine="not_a_quam")

    def test_get_backend_machine_is_set(self, qm_provider):
        backend = qm_provider.get_backend()
        assert backend.machine is not None
        assert hasattr(backend.machine, "active_qubits")
