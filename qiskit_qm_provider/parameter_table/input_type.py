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

"""InputType and Direction enums for parameter/stream configuration (e.g. OPNIC, INPUT_STREAM).

Author: Arthur Strauss
Date: 2026-02-08
"""

from enum import Enum
from typing import Optional


class InputType(Enum):
    """How parameter values are delivered to the OPX during program execution."""

    OPNIC = "OPNIC"
    INPUT_STREAM = "INPUT_STREAM"
    IO1 = "IO1"
    IO2 = "IO2"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: Optional[str]) -> Optional["InputType"]:
        """Parse an input type from its string value.

        Args:
            value: Enum value string (e.g. ``\"INPUT_STREAM\"``) or ``None``.

        Returns:
            Matching :class:`InputType`, or ``None`` when ``value`` is ``None``.

        Raises:
            ValueError: If the string does not match a known input type.
        """
        if value is None:
            return None
        for input_type in cls:
            if input_type.value == value:
                return input_type
        raise ValueError(f"Invalid input type: {value}")


class Direction(Enum):
    """Data-flow direction for OPNIC packet streams, expressed from the **QUA program's**
    perspective and aligned 1:1 with Quarc's ``Direction``.

    * **INCOMING** — into the QUA program (classical/OPNIC -> OPX); the ``rcv`` /
      ``push_to_opx`` direction.
    * **OUTGOING** — out of the QUA program (OPX -> classical/OPNIC); the
      ``stream_back`` / ``fetch_from_opx`` direction.
    * **BOTH** — bidirectional.
    """

    INCOMING = "INCOMING"  # into QUA (classical/OPNIC -> OPX)
    OUTGOING = "OUTGOING"  # out of QUA (OPX -> classical/OPNIC)
    BOTH = "BOTH"  # bidirectional

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: Optional[str]) -> Optional["Direction"]:
        if value is None:
            return None
        for direction in cls:
            if direction.value == value:
                return direction
        raise ValueError(f"Invalid direction: {value}")
