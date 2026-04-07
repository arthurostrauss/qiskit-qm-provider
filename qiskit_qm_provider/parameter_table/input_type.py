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
    OPNIC = "OPNIC"
    INPUT_STREAM = "INPUT_STREAM"
    IO1 = "IO1"
    IO2 = "IO2"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: Optional[str]) -> Optional["InputType"]:
        if value is None:
            return None
        for input_type in cls:
            if input_type.value == value:
                return input_type
        raise ValueError(f"Invalid input type: {value}")


class Direction(Enum):
    """
    The direction of the data flow for OPNIC packet streams.
    INCOMING: OPX -> classical (OPNIC)
    OUTGOING: classical (OPNIC) -> OPX
    BOTH: bidirectional.
    """

    INCOMING = "INCOMING"  # OPX -> OPNIC
    OUTGOING = "OUTGOING"  # OPNIC -> OPX
    BOTH = "BOTH"  # OPNIC <-> OPX

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
