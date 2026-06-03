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

"""Pulse integration for Qiskit 1.x (legacy).

Supports converting **gate pulse schedules** to QUA via ``schedule_to_qua_macro``.
Qiskit Pulse **measurement instructions** are not supported; use circuit-level
``measure`` gates and :func:`~qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes`
in hybrid programs instead.
"""

__all__: list[str] = []

try:
    from .pulse_support_utils import (
        handle_parameterized_channel,
        _handle_parameterized_instruction,
        schedule_to_qua_macro,
        validate_schedule,
    )

    __all__.extend(
        [
            "handle_parameterized_channel",
            "_handle_parameterized_instruction",
            "schedule_to_qua_macro",
            "validate_schedule",
        ]
    )
except ImportError:
    pass

try:
    from .quam_qiskit_pulse import FluxChannel, QuAMQiskitPulse

    __all__.extend(["FluxChannel", "QuAMQiskitPulse"])
except ImportError:
    pass
