from enum import Enum
from typing import Optional, Union, ClassVar


class InputType(Enum):
    DGX_Q = "DGX_Q"
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
    The direction of the data flow.
    INCOMING: OPX -> DGX_Q
    OUTGOING: DGX_Q -> OPX

    Useful for defining DGX Quantum streams.
    """

    INCOMING = "INCOMING"  # OPX -> DGX_Q
    OUTGOING = "OUTGOING"  # DGX_Q -> OPX
    BOTH = "BOTH"  # DGX_Q -> OPX and OPX -> DGX_Q

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
