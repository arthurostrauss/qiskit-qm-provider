"""Tests for QmSaasProvider's connection to the QM cloud simulator (no network access)."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("qm_saas")

from qiskit_qm_provider import QmSaasProvider


@pytest.fixture
def provider():
    with patch("qm_saas.QmSaas") as mock_client_cls:
        client = mock_client_cls.return_value
        client.latest_version.return_value = "latest"
        yield QmSaasProvider(email="user@example.com", password="secret", host="qm-saas.example.com")


def test_get_backend_uses_qmm_connection_params(provider):
    """The QMM must be built from ``qmm_connection_params`` so it carries the TLS credentials."""
    params = {
        "host": "sim.example.com",
        "port": 443,
        "connection_headers": {"id": "abc", "token": "xyz"},
        "credentials": object(),
        "follow_gateway_redirections": False,
    }
    provider.instance.qmm_connection_params = params
    quam_cls = MagicMock()
    backend_cls = MagicMock()

    with patch("qm.QuantumMachinesManager") as mock_qmm_cls:
        backend = provider.get_backend(quam_cls=quam_cls, backend_cls=backend_cls)

    provider.instance.spawn.assert_called_once()
    mock_qmm_cls.assert_called_once_with(**params)
    _, kwargs = backend_cls.call_args
    assert kwargs["qmm"] is mock_qmm_cls.return_value
    assert kwargs["provider"] is provider
    assert backend is backend_cls.return_value


def test_qm_saas_exposes_qmm_connection_params():
    """Guard the ``qm-saas>=1.2.0`` floor: older releases lack ``qmm_connection_params``."""
    from qm_saas import QmSaasInstance

    assert isinstance(getattr(QmSaasInstance, "qmm_connection_params", None), property)
