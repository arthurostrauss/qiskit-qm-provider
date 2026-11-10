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

"""Fixed-point arithmetic mirroring QUA's ``fixed`` type, including overflow.

:class:`FixedPoint` reproduces the controller's 4.28 ``fixed`` arithmetic on the host.
Like QUA (checked on hardware), **overflow wraps around** in two's complement: the
result of an operation is reduced modulo ``2**bit_width``, it is *not* clamped. With
the default 4.28 format the representable range is ``[-8, 8)``, so ``7 + 3`` gives
``-6`` and ``3 * 3`` gives ``-7``. Whenever an operation (or a conversion from a float)
leaves the representable range, a :class:`FixedPointOverflowWarning` is emitted and the
wrapped value is returned, so host replays stay bit-exact with the controller while
the overflow is still visible.

Author: Arthur Strauss
Date: 2026-02-08
"""

import warnings


class FixedPointOverflowWarning(RuntimeWarning):
    """A :class:`FixedPoint` operation overflowed and wrapped around, as QUA does.

    Filter or escalate it with the standard :mod:`warnings` machinery, e.g.
    ``warnings.simplefilter("error", FixedPointOverflowWarning)``.
    """


class FixedPointApproximationWarning(RuntimeWarning):
    """A :class:`FixedPoint` operation is only an approximation of the controller's result.

    Emitted by division, whose hardware implementation is approximate (relative error up
    to about ``1e-6``) while the host computes the exact floored quotient.
    """


class FixedPoint:
    """Fixed-point numeric type mirroring QUA's ``fixed`` arithmetic.

    Values are stored as signed integers scaled by ``2**fractional_bits`` within a
    configurable bit width (default: 32 bits, 28 fractional, i.e. range ``[-8, 8)``).

    Overflow follows QUA: results wrap around modulo ``2**bit_width`` and a
    :class:`FixedPointOverflowWarning` is emitted. Multiplication and right shifts
    truncate toward negative infinity (arithmetic shift), as the controller does.

    Checked bit for bit on hardware: ``+``, ``-``, ``*``, ``<<`` and ``>>`` (including
    overflow). **Division is not bit-exact**: the controller computes an approximation
    (relative error up to about ``1e-6``, i.e. a few tens of LSB in 4.28), whereas
    :meth:`__truediv__` returns the exact floored quotient, and a quotient that overflows
    is not reproduced. ``&`` and ``|`` have not been compared with the controller.
    """

    def __init__(self, value, fractional_bits=28, bit_width=32):
        """Create a fixed-point value from a float or integer scale.

        The value is rounded to the nearest representable fixed-point number, as
        QUA does for ``declare(fixed, value=...)`` (verified on hardware). A value
        outside the representable range wraps around and emits a
        :class:`FixedPointOverflowWarning` (QUA programs should not declare such
        constants).

        Args:
            value: Initial value (float interpreted in fixed-point units).
            fractional_bits: Number of fractional bits in the representation.
            bit_width: Total signed bit width including the sign bit.
        """
        self.fractional_bits = fractional_bits
        self.scale = 1 << fractional_bits
        self.bit_width = bit_width
        self.max_value = (1 << (bit_width - 1)) - 1
        self.min_value = -(1 << (bit_width - 1))
        self.value = self._wrap(round(value * self.scale))

    def _wrap(self, value, stacklevel=3):
        """Reduce a raw integer to the signed ``bit_width`` range, warning on overflow."""
        if self.min_value <= value <= self.max_value:
            return value
        warnings.warn(
            f"FixedPoint overflow: raw value {value} is outside [{self.min_value}, {self.max_value}] "
            f"(about {value / self.scale:g} for a representable range of "
            f"[{self.min_value / self.scale:g}, {(self.max_value + 1) / self.scale:g})). "
            "The result wraps around, as in QUA.",
            FixedPointOverflowWarning,
            stacklevel=stacklevel,
        )
        modulus = 1 << self.bit_width
        value &= modulus - 1
        return value - modulus if value > self.max_value else value

    def _new(self, raw):
        """Build a result from a raw scaled integer (wrapping it if needed)."""
        out = FixedPoint.__new__(FixedPoint)
        out.fractional_bits = self.fractional_bits
        out.scale = self.scale
        out.bit_width = self.bit_width
        out.max_value = self.max_value
        out.min_value = self.min_value
        out.value = self._wrap(raw, stacklevel=4)
        return out

    def __add__(self, other):
        if isinstance(other, FixedPoint):
            result = self.value + other.value
        elif isinstance(other, int):
            result = self.value + (other << self.fractional_bits)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'FixedPoint' and '{}'".format(type(other).__name__))
        return self._new(result)

    def __sub__(self, other):
        if isinstance(other, FixedPoint):
            result = self.value - other.value
        elif isinstance(other, int):
            result = self.value - (other << self.fractional_bits)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'FixedPoint' and '{}'".format(type(other).__name__))
        return self._new(result)

    def __mul__(self, other):
        if isinstance(other, FixedPoint):
            result = (self.value * other.value) >> self.fractional_bits
        elif isinstance(other, int):
            result = self.value * other
        else:
            raise TypeError("Unsupported operand type(s) for *: 'FixedPoint' and '{}'".format(type(other).__name__))
        return self._new(result)

    def __truediv__(self, other):
        """Exact floored quotient, with a :class:`FixedPointApproximationWarning`.

        The controller's division is only approximate (see the class docs), so the host
        result can differ from it by a few tens of LSB, and quotients that overflow are
        not reproduced.
        """
        warnings.warn(
            "FixedPoint division is exact on the host but only approximate on the controller "
            "(relative error up to ~1e-6); results will not match QUA bit for bit.",
            FixedPointApproximationWarning,
            stacklevel=2,
        )
        if isinstance(other, FixedPoint):
            result = (self.value << self.fractional_bits) // other.value
        elif isinstance(other, int):
            result = self.value // other
        else:
            raise TypeError("Unsupported operand type(s) for /: 'FixedPoint' and '{}'".format(type(other).__name__))
        return self._new(result)

    def __lshift__(self, other):
        return self._new(self.value << other)

    def __rshift__(self, other):
        return self._new(self.value >> other)

    def __and__(self, other):
        if isinstance(other, FixedPoint):
            result = self.value & other.value
        elif isinstance(other, int):
            result = self.value & other
        else:
            raise TypeError("Unsupported operand type(s) for &: 'FixedPoint' and '{}'".format(type(other).__name__))
        return self._new(result)

    def __or__(self, other):
        if isinstance(other, FixedPoint):
            result = self.value | other.value
        elif isinstance(other, int):
            result = self.value | other
        else:
            raise TypeError("Unsupported operand type(s) for |: 'FixedPoint' and '{}'".format(type(other).__name__))
        return self._new(result)

    def __repr__(self):
        return f"{self.value / self.scale:.10f}"

    def to_int(self) -> int:
        """Return the integer part, floored toward negative infinity (arithmetic shift)."""
        return self.value >> self.fractional_bits

    def to_unsafe_int(self) -> int:
        """Return the raw scaled integer without fractional-bit shifting."""
        return self.value

    def to_float(self) -> float:
        """Return the value as a Python float."""
        return self.value / self.scale

    def __float__(self) -> float:
        """Support ``float(fixed_point)`` by delegating to :meth:`to_float`."""
        return self.to_float()

    @classmethod
    def from_int(cls, int_value, fractional_bits=28, bit_width=32):
        """Construct a :class:`FixedPoint` from an integer fixed-point representation.

        Args:
            int_value: Scaled integer value (already in fixed-point units).
            fractional_bits: Number of fractional bits.
            bit_width: Total signed bit width.

        Returns:
            New :class:`FixedPoint` instance.
        """
        out = cls(0, fractional_bits, bit_width)
        out.value = out._wrap(int(int_value))
        return out
