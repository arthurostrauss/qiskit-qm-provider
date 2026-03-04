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

"""Fixed-point arithmetic for QUA-compatible numeric representation.

Author: Arthur Strauss
Date: 2026-02-08
"""


class FixedPoint:
    def __init__(self, value, fractional_bits=28, bit_width=32):
        self.fractional_bits = fractional_bits
        self.scale = 1 << fractional_bits
        self.bit_width = bit_width
        self.max_value = (1 << (bit_width - 1)) - 1
        self.min_value = -(1 << (bit_width - 1))
        self.value = self._saturate(int(value * self.scale))

    def _saturate(self, value):
        if value > self.max_value:
            return self.max_value
        elif value < self.min_value:
            return self.min_value
        else:
            return value

    def __add__(self, other):
        if isinstance(other, FixedPoint):
            result = self.value + other.value
        elif isinstance(other, int):
            result = self.value + (other << self.fractional_bits)
        else:
            raise TypeError(
                "Unsupported operand type(s) for +: 'FixedPoint' and '{}'".format(
                    type(other).__name__
                )
            )
        return FixedPoint(
            self._saturate(result) / self.scale, self.fractional_bits, self.bit_width
        )

    def __sub__(self, other):
        if isinstance(other, FixedPoint):
            result = self.value - other.value
        elif isinstance(other, int):
            result = self.value - (other << self.fractional_bits)
        else:
            raise TypeError(
                "Unsupported operand type(s) for -: 'FixedPoint' and '{}'".format(
                    type(other).__name__
                )
            )
        return FixedPoint(
            self._saturate(result) / self.scale, self.fractional_bits, self.bit_width
        )

    def __mul__(self, other):
        if isinstance(other, FixedPoint):
            result = (self.value * other.value) >> self.fractional_bits
        elif isinstance(other, int):
            result = self.value * other
        else:
            raise TypeError(
                "Unsupported operand type(s) for *: 'FixedPoint' and '{}'".format(
                    type(other).__name__
                )
            )
        return FixedPoint(
            self._saturate(result) / self.scale, self.fractional_bits, self.bit_width
        )

    def __truediv__(self, other):
        if isinstance(other, FixedPoint):
            result = (self.value << self.fractional_bits) // other.value
        elif isinstance(other, int):
            result = self.value // other
        else:
            raise TypeError(
                "Unsupported operand type(s) for /: 'FixedPoint' and '{}'".format(
                    type(other).__name__
                )
            )
        return FixedPoint(
            self._saturate(result) / self.scale, self.fractional_bits, self.bit_width
        )

    def __lshift__(self, other):
        result = self.value << other
        return FixedPoint(
            self._saturate(result) / self.scale, self.fractional_bits, self.bit_width
        )

    def __rshift__(self, other):
        result = self.value >> other
        return FixedPoint(
            self._saturate(result) / self.scale, self.fractional_bits, self.bit_width
        )

    def __and__(self, other):
        if isinstance(other, FixedPoint):
            result = self.value & other.value
        elif isinstance(other, int):
            result = self.value & other
        else:
            raise TypeError(
                "Unsupported operand type(s) for &: 'FixedPoint' and '{}'".format(
                    type(other).__name__
                )
            )
        return FixedPoint(result / self.scale, self.fractional_bits, self.bit_width)

    def __or__(self, other):
        if isinstance(other, FixedPoint):
            result = self.value | other.value
        elif isinstance(other, int):
            result = self.value | other
        else:
            raise TypeError(
                "Unsupported operand type(s) for |: 'FixedPoint' and '{}'".format(
                    type(other).__name__
                )
            )
        return FixedPoint(result / self.scale, self.fractional_bits, self.bit_width)

    def __repr__(self):
        return f"{self.value / self.scale:.10f}"

    def to_int(self) -> int:
        return self.value >> self.fractional_bits

    def to_unsafe_int(self) -> int:
        return self.value

    def to_float(self) -> float:
        return self.value / self.scale

    @classmethod
    def from_int(cls, int_value, fractional_bits=28, bit_width=32):
        return cls(int_value / (1 << fractional_bits), fractional_bits, bit_width)
