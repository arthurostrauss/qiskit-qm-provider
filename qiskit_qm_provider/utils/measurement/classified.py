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

"""Reject measurement levels the provider does not assemble.

Author: Arthur Strauss
Date: 2026-10-29
"""


def require_classified_meas_level(meas_level, *, context: str = "") -> None:
    """Raise if ``meas_level`` is not classified 0/1 readout.

    Only :data:`~qiskit.result.models.MeasLevel.CLASSIFIED` is supported end-to-end
    for sampler and ``backend.run()`` result assembly.
    """
    from qiskit.result.models import MeasLevel

    if meas_level != MeasLevel.CLASSIFIED:
        suffix = f" ({context})" if context else ""
        raise NotImplementedError(
            f"Only MeasLevel.CLASSIFIED measurement is supported{suffix}; got {meas_level!r}."
        )
