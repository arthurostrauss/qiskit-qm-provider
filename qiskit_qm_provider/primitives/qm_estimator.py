from typing import Iterable

from qiskit.primitives import (
    BaseEstimatorV2,
    EstimatorPubLike,
    BasePrimitiveJob,
    PrimitiveResult,
    PubResult,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from dataclasses import dataclass

from ..backend.qm_backend import QMBackend


class QMEstimatorV2(BaseEstimatorV2):
    """QM Estimator V2 class for Qiskit Quantum Machine backend."""

    def run(self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None):
        raise NotImplementedError("QMEstimatorV2 not implemented yet. ")

    def __init__(self, backend: QMBackend, **kwargs):
        self._backend = backend

    @property
    def backend(self) -> QMBackend:
        """Return the QM backend associated with this estimator."""
        return self._backend
