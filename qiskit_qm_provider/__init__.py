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

"""Qiskit-QM provider package: backends, primitives, and utilities for Quantum Machines hardware.

Author: Arthur Strauss
Date: 2026-02-08
"""

import importlib.util
import warnings

from .backend import (
    QMBackend,
    QISKIT_PULSE_AVAILABLE,
    FluxTunableTransmonBackend,
    QMInstructionProperties,
)
from .providers.qm_provider import QMProvider

# Probe availability without importing quarc (``qiskit_qm_module`` pulls quarc in).
QUARC_AVAILABLE = importlib.util.find_spec("quarc") is not None

if not QUARC_AVAILABLE:
    warnings.warn(
        "The `quarc` package is not available; `QiskitQMModule` and related Quarc "
        "integration features are not loaded. Install `quarc` to use them.",
        ImportWarning,
    )

if QISKIT_PULSE_AVAILABLE:
    from .pulse.quam_qiskit_pulse import QuAMQiskitPulse, FluxChannel
from .fixed_point import FixedPoint
from .parameter_table import *
from .additional_gates import *
from .primitives.qm_sampler import QMSamplerV2, QMSamplerOptions
from .primitives.qm_estimator import QMEstimatorV2, QMEstimatorOptions
from .backend.backend_utils import (
    get_measurement_outcomes,
    add_basic_macros,
    assign_struct_with_table,
    dump_qua_script,
    get_qua_script,
    pack_register_to_int,
)
from .backend.qua_circuit_compilation import (
    QuaCircuitCompilation,
    MeasurementOutcomeTable,
)
from .backend.measurement_field import MeasurementRegisterField

__all__ = [
    "QMBackend",
    "FluxTunableTransmonBackend",
    "QMInstructionProperties",
    "FixedPoint",
    "ParameterTable",
    "Parameter",
    "ParameterPool",
    "Direction",
    "InputType",
    "QUA2DArray",
    "QMSamplerV2",
    "QMSamplerOptions",
    "QMEstimatorV2",
    "QMEstimatorOptions",
    "get_measurement_outcomes",
    "add_basic_macros",
    "assign_struct_with_table",
    "dump_qua_script",
    "get_qua_script",
    "pack_register_to_int",
    "QuaCircuitCompilation",
    "MeasurementRegisterField",
    "MeasurementOutcomeTable",
    "QMProvider",
    "QUARC_AVAILABLE",
]

if QUARC_AVAILABLE:
    __all__.append("QiskitQMModule")

if QISKIT_PULSE_AVAILABLE:
    try:
        from .pulse.quam_qiskit_pulse import QuAMQiskitPulse, FluxChannel
    except ImportError:
        pass
    else:
        __all__.extend(["QuAMQiskitPulse", "FluxChannel"])

try:
    from .providers.qm_saas_provider import QmSaasProvider

    __all__.append("QmSaasProvider")
except ImportError:
    pass

try:
    from .providers.iqcc_cloud_provider import IQCCProvider

    __all__.append("IQCCProvider")
except ImportError:
    pass


def __getattr__(name: str):
    """Lazy-load Quarc-backed symbols so ``import qiskit_qm_provider`` stays quarc-free."""
    if name == "QiskitQMModule":
        if not QUARC_AVAILABLE:
            raise ImportError(
                "QiskitQMModule requires the `quarc` package. Install `quarc` to use it."
            ) from None
        from .qiskit_qm_module import QiskitQMModule

        globals()["QiskitQMModule"] = QiskitQMModule
        return QiskitQMModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
