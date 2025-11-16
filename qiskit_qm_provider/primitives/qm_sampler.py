from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Iterable, Literal, Optional

from qiskit.primitives import (
    BaseSamplerV2,
    SamplerPubLike,
)
from dataclasses import dataclass

from qiskit.primitives.containers.sampler_pub import SamplerPub
from ..job.qm_sampler_job import QMSamplerJob, IQCCSamplerJob

from ..backend.backend_utils import validate_circuits
from ..parameter_table import InputType
from ..backend.qm_backend import QMBackend
from qiskit.result.models import MeasLevel, MeasReturnType
from qm import QuantumMachinesManager

meas_level_dict = {
    "classified": MeasLevel.CLASSIFIED,
    "kerneled": MeasLevel.KERNELED,
    "avg_kerneled": MeasLevel.KERNELED,
}
meas_return_type_dict = {"kerneled": MeasReturnType.SINGLE, "avg_kerneled": MeasReturnType.AVERAGE}


@dataclass
class QMSamplerOptions:
    """Options for :class:`~.QMSamplerV2`"""

    default_shots: int = 1024
    """The default shots to use if none are specified in :meth:`~.run`.
    Default: 1024.
    """

    input_type: Optional[InputType] = None
    """The input mechanism to load the parameter values to the OPX. Choices are:
    - :class:`~.InputType.INPUT_STREAM`: Input stream mechanism.
    - :class:`~.InputType.IO1`: IO1.
    - :class:`~.InputType.IO2`: IO2.
    - :class:`~.InputType.DGX_Q`: Using DGX Quantum communication.
    - None: Preload at compile time the parameter values to the OPX (Warning: This should be used only for small number of parameters)
    Default: None."""

    run_options: dict[str, Any] | None = None
    """A dictionary of options to pass to the backend's ``run()`` method.
    Default: None (no option passed to backend's ``run`` method)
    """
    meas_level: Literal["classified", "kerneled", "avg_kerneled"] = "classified"

    def __post_init__(self):
        if isinstance(self.input_type, str):
            self.input_type = InputType(self.input_type)
        if not isinstance(self.input_type, InputType):
            raise TypeError(f"input_type must be of type InputType, got {type(self.input_type)}")
        if self.run_options is not None and not isinstance(self.run_options, dict):
            raise TypeError(f"run_options must be a dictionary, got {type(self.run_options)}")


class QMSamplerV2(BaseSamplerV2):
    """QM Sampler class."""

    def __init__(self, backend: QMBackend, options: QMSamplerOptions | dict | None = None):
        self._backend = backend
        self._options = QMSamplerOptions(**options) if isinstance(options, dict) else options or QMSamplerOptions()

    @property
    def options(self) -> QMSamplerOptions:
        """Return the options"""
        return self._options

    @property
    def backend(self) -> QMBackend:
        """Return the backend"""
        return self._backend

    def run(self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None) -> QMSamplerJob:
        if shots is None:
            shots = self._options.default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        coerced_pubs = self._validate_pubs(coerced_pubs)
        job_obj = QMSamplerJob if isinstance(self.backend.qmm, QuantumMachinesManager) else IQCCSamplerJob
        backend_options = deepcopy(self.backend.options.__dict__)

        backend_options["meas_level"] = meas_level_dict[self._options.meas_level]
        backend_options["meas_return_type"] = meas_return_type_dict.get(self._options.meas_level, MeasReturnType.SINGLE)
        backend_options["shots"] = shots
        backend_options.update(self._options.run_options or {})

        job = job_obj(self.backend, coerced_pubs, self.options.input_type, **backend_options)
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
