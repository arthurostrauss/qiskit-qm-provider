from __future__ import annotations

from typing import Iterable, Literal, Any

from qiskit.circuit.classical import types
from qiskit.primitives import (
    BaseEstimatorV2,
    EstimatorPubLike,
    BasePrimitiveJob,
    PrimitiveResult,
    PubResult,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from dataclasses import dataclass

from qiskit.transpiler import PassManagerConfig, PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

from ..parameter_table import InputType

from ..backend.qm_backend import QMBackend


@dataclass
class QMEstimatorOptions:
    """Options for :class:`~.QMEstimatorV2`."""

    default_precision: float = 0.015625
    """The default precision to use if none are specified in :meth:`~run`.
    Default: 0.015625 (1 / sqrt(4096)).
    """

    abelian_grouping: bool = True
    """Whether the observables should be grouped into sets of qubit-wise commuting observables.
    Default: True.
    """

    input_type: InputType | Literal["INPUT_STREAM", "IO1", "IO2", "DGX"] = InputType.INPUT_STREAM
    """The input mechanism to load the parameter values to the OPX. Choices are:
    - :class:`~.InputType.INPUT_STREAM`: Input stream mechanism.
    - :class:`~.InputType.IO1`: IO1.
    - :class:`~.InputType.IO2`: IO2.
    - :class:`~.InputType.DGX`: Using DGX Quantum communication.
    Default: InputType.INPUT_STREAM."""

    run_options: dict[str, Any] | None = None
    """A dictionary of options to pass to the backend's ``run()`` method.
    Default: None (no option passed to backend's ``run`` method)
    """

    def __post_init__(self):
        if isinstance(self.input_type, str):
            self.input_type = InputType(self.input_type)
        if not isinstance(self.input_type, InputType):
            raise TypeError(f"input_type must be of type InputType, got {type(self.input_type)}")
        if self.run_options is not None and not isinstance(self.run_options, dict):
            raise TypeError(f"run_options must be a dictionary, got {type(self.run_options)}")


class QMEstimatorV2(BaseEstimatorV2):
    """QM Estimator V2 class for Qiskit Quantum Machine backend."""

    def __init__(self, backend: QMBackend, options: QMEstimatorOptions | dict | None = None):
        self._backend = backend
        self._options = QMEstimatorOptions(**options) if isinstance(options, dict) else options or QMEstimatorOptions()

        basis = PassManagerConfig.from_backend(backend).basis_gates
        opt1q = Optimize1qGatesDecomposition(basis=basis, target=backend.target)

        self._passmanager = PassManager([opt1q])

    def run(self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None):
        """Run the estimator on the given PUBs."""
        if precision is None:
            precision = self.options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)

        # Modify all quantum circuits in the PUBs to insert switch statements
        for pub in coerced_pubs:
            qc = pub.circuit
            observables_vars = [qc.add_input(f"obs_{i}", types.Uint(4)) for i in range(qc.num_qubits)]
            for q, qubit in enumerate(qc.qubits):
                with qc.switch(observables_vars[q]) as case:
                    with case(0):
                        qc.delay(16, qubit)
                    with case(1):
                        qc.h(qubit)
                    with case(2):
                        qc.sdg(qubit)
                        qc.h(qubit)
                qc.measure_all()

    @property
    def backend(self) -> QMBackend:
        """Return the QM backend associated with this estimator."""
        return self._backend

    @property
    def options(self) -> QMEstimatorOptions:
        """Return the options for this estimator."""
        return self._options

    def _validate_pubs(self, pubs: list[EstimatorPub]):
        for i, pub in enumerate(pubs):
            if pub.precision <= 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than or equal to 0 ({pub.precision}). ",
                    "But precision should be larger than 0.",
                )
