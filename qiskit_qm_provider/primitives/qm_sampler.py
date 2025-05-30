from __future__ import annotations

import warnings
from typing import Any, Iterable, Literal

from iqcc_cloud_client import IQCC_Cloud
from qiskit.primitives import (
    BaseSamplerV2,
    SamplerPubLike,
)
from dataclasses import dataclass

from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit_qm_provider.job.qm_sampler_job import QMPrimitiveJob, IQCCPrimitiveJob

from qiskit_qm_provider.backend.backend_utils import _QASM3_DUMP_LOOSE_BIT_PREFIX, validate_circuits

from qiskit_qm_provider.parameter_table import InputType, ParameterTable
from qiskit_qm_provider.backend.qm_backend import QMBackend

# from .qm_sampler_job import QMPrimitiveJob
from quam.utils.qua_types import QuaScalar


@dataclass
class QMSamplerOptions:
    """Options for :class:`~.QMSamplerV2`"""

    default_shots: int = 1024
    """The default shots to use if none are specified in :meth:`~.run`.
    Default: 1024.
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


class QMSamplerV2(BaseSamplerV2):
    """QM Sampler class."""

    def __init__(self, backend: QMBackend, options: QMSamplerOptions | None = None):

        self._backend = backend
        self._options = options or QMSamplerOptions()

    @property
    def options(self) -> QMSamplerOptions:
        """Return the options"""
        return self._options

    @property
    def backend(self) -> QMBackend:
        """Return the backend"""
        return self._backend

    def run(self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None) -> QMPrimitiveJob:
        if shots is None:
            shots = self._options.default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        coerced_pubs = self._validate_pubs(coerced_pubs)
        job_obj = IQCCPrimitiveJob if isinstance(self.backend.qmm, IQCC_Cloud) else QMPrimitiveJob
        job = job_obj(self.backend, coerced_pubs, self._options.input_type)
        job.submit()
        return job

    def _validate_pubs(self, pubs: list[SamplerPub]):
        for i, pub in enumerate(pubs):
            if len(pub.circuit.cregs) == 0:
                warnings.warn(
                    f"The {i}-th pub's circuit has no output classical registers and so the result "
                    "will be empty. Did you mean to add measurement instructions?",
                    UserWarning,
                )
        new_circuits = validate_circuits(
            [pub.circuit for pub in pubs],
            should_reset=not self._backend.options.skip_reset,
            check_for_params=False,
        )
        new_pubs = [
            SamplerPub(circuit, shots=pub.shots, parameter_values=pub.parameter_values)
            for circuit, pub in zip(new_circuits, pubs)
        ]
        return new_pubs
