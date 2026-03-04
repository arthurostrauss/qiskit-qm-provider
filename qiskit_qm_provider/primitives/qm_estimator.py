# Copyright 2026 Arthur Strauss
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QMEstimatorV2: Qiskit Estimator primitive for QM backends (expectation values from QUA).

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Literal, Any, Optional, Union

from qiskit.circuit.classical import types
from qiskit.primitives import (
    BaseEstimatorV2,
    EstimatorPubLike,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from dataclasses import asdict, dataclass

from qiskit.transpiler import PassManagerConfig, PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from ..parameter_table import InputType

from ..backend.qm_backend import QMBackend
from ..backend.backend_utils import validate_circuits, logically_active_qubits


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

    input_type: Optional[
        Union[InputType, Literal["INPUT_STREAM", "IO1", "IO2", "DGX_Q"]]
    ] = None
    """The input mechanism to load the parameter values to the OPX. Choices are:
    - :class:`~.InputType.INPUT_STREAM`: Input stream mechanism.
    - :class:`~.InputType.IO1`: IO1.
    - :class:`~.InputType.IO2`: IO2.
    - :class:`~.InputType.DGX_Q`: Using DGX Quantum communication.
    - None: Preload at compile time the parameter values to the OPX.
    Default: None."""

    run_options: dict[str, Any] | None = None
    """A dictionary of options to pass to the backend's ``run()`` method.
    Default: None (no option passed to backend's ``run`` method)
    """

    def __post_init__(self):
        if isinstance(self.input_type, str):
            self.input_type = InputType(self.input_type)
        if self.input_type is not None and not isinstance(self.input_type, InputType):
            raise TypeError(
                f"input_type must be of type InputType, got {type(self.input_type)}"
            )
        if self.run_options is not None and not isinstance(self.run_options, dict):
            raise TypeError(
                f"run_options must be a dictionary, got {type(self.run_options)}"
            )

    def as_dict(self) -> dict:
        return asdict(self)


class QMEstimatorV2(BaseEstimatorV2):
    """QM Estimator V2 class for Qiskit Quantum Machine backend."""

    def __init__(
        self, backend: QMBackend, options: QMEstimatorOptions | dict | None = None
    ):
        self._backend = backend
        self._options = (
            QMEstimatorOptions(**options)
            if isinstance(options, dict)
            else options or QMEstimatorOptions()
        )
        self._job = None

        basis = PassManagerConfig.from_backend(backend).basis_gates
        opt1q = Optimize1qGatesDecomposition(basis=basis, target=backend.target)

        self._passmanager = PassManager([opt1q])
        qc_switch_obs = QuantumCircuit(1)
        obs_var = qc_switch_obs.add_input("obs", types.Uint(4))
        with qc_switch_obs.switch(obs_var) as case_obs:
            with case_obs(1):
                qc_switch_obs.h(0)
            with case_obs(2):
                qc_switch_obs.sdg(0)
                qc_switch_obs.h(0)
            with case_obs(case_obs.DEFAULT):
                qc_switch_obs.id(0)

        qc_switch_obs = self._passmanager.run(qc_switch_obs)
        self._switch_obs_circuit = qc_switch_obs

    def run(self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None):
        """Run the estimator on the given PUBs."""
        if precision is None:
            precision = self.options.default_precision
        pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        pubs = self.validate_estimator_pubs(pubs)
        # Update Target of backend if needed
        self.backend.update_target(self.options.input_type)
        from ..job.qm_estimator_job import QMEstimatorJob, IQCCEstimatorJob
        from qm import QuantumMachinesManager

        job_obj = (
            QMEstimatorJob
            if isinstance(self.backend.qmm, QuantumMachinesManager)
            else IQCCEstimatorJob
        )
        backend_options = deepcopy(self.backend.options.__dict__)
        backend_options.update(self._options.run_options or {})
        job = job_obj(
            self._backend,
            pubs,
            self.options.input_type,
            switch_obs_circuit=self._switch_obs_circuit,
            run_options=backend_options,
            abelian_grouping=self.options.abelian_grouping,
            default_precision=precision,
        )
        self._job = job
        job.submit()
        return job

    @property
    def backend(self) -> QMBackend:
        """Return the QM backend associated with this estimator."""
        return self._backend

    @property
    def options(self) -> QMEstimatorOptions:
        """Return the options for this estimator."""
        return self._options

    def validate_estimator_pubs(self, pubs: list[EstimatorPub]) -> list[EstimatorPub]:
        new_pubs = []
        for i, pub in enumerate(pubs):
            if pub.precision <= 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than or equal to 0 ({pub.precision}). ",
                    "But precision should be larger than 0.",
                )
            if (
                pub.circuit.num_qubits != self._backend.num_qubits
                or pub.observables.num_qubits != self._backend.num_qubits
            ):
                raise ValueError(
                    f"The {i}-th pub has {pub.circuit.num_qubits} circuit qubits and {pub.observables.num_qubits} observables qubits, but the backend has {self._backend.num_qubits} qubits.",
                    "Make sure you have transpiled the circuit to the backend's target as well as applied the circuit layout to the observables.",
                )

            qc = pub.circuit.copy()
            active_qubits = logically_active_qubits(pub.circuit)
            qubit_indices = [qc.find_bit(q).index for q in active_qubits]
            num_active_qubits = len(active_qubits)
            creg = ClassicalRegister(num_active_qubits, name="__c")
            qc.add_register(creg)
            for q, qubit in enumerate(active_qubits):
                qc.compose(
                    self._switch_obs_circuit,
                    [qubit],
                    inplace=True,
                    var_remap={"obs": f"obs_{q}"},
                )
            qc.measure(active_qubits, creg)
            qc = validate_circuits(qc)[0]

            new_pub = EstimatorPub(
                qc, pub.observables, pub.parameter_values, pub.precision
            )
            new_pubs.append(new_pub)

        return new_pubs
