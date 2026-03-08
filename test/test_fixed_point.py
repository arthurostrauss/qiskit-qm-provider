"""Tests for FixedPoint arithmetic."""

import pytest
from qiskit_qm_provider.fixed_point import FixedPoint


class TestFixedPointInit:
    def test_default_params(self):
        fp = FixedPoint(0.5)
        assert fp.fractional_bits == 28
        assert fp.bit_width == 32
        assert fp.scale == 1 << 28

    def test_custom_fractional_bits(self):
        fp = FixedPoint(0.5, fractional_bits=16, bit_width=32)
        assert fp.fractional_bits == 16
        assert fp.scale == 1 << 16

    def test_zero(self):
        fp = FixedPoint(0.0)
        assert fp.to_float() == 0.0
        assert fp.to_int() == 0

    def test_positive_value(self):
        fp = FixedPoint(1.5)
        assert abs(fp.to_float() - 1.5) < 1e-7

    def test_negative_value(self):
        fp = FixedPoint(-0.25)
        assert abs(fp.to_float() - (-0.25)) < 1e-7

    def test_saturation_max(self):
        fp = FixedPoint(100.0, fractional_bits=28, bit_width=32)
        assert fp.value == fp.max_value

    def test_saturation_min(self):
        fp = FixedPoint(-100.0, fractional_bits=28, bit_width=32)
        assert fp.value == fp.min_value


class TestFixedPointArithmetic:
    def test_add_fixed_points(self):
        a = FixedPoint(0.5)
        b = FixedPoint(0.25)
        result = a + b
        assert abs(result.to_float() - 0.75) < 1e-7

    def test_add_int(self):
        a = FixedPoint(0.5)
        result = a + 1
        assert abs(result.to_float() - 1.5) < 1e-7

    def test_add_unsupported_type(self):
        a = FixedPoint(0.5)
        with pytest.raises(TypeError):
            a + "string"

    def test_sub_fixed_points(self):
        a = FixedPoint(0.75)
        b = FixedPoint(0.25)
        result = a - b
        assert abs(result.to_float() - 0.5) < 1e-7

    def test_sub_int(self):
        a = FixedPoint(1.5)
        result = a - 1
        assert abs(result.to_float() - 0.5) < 1e-7

    def test_sub_unsupported_type(self):
        a = FixedPoint(0.5)
        with pytest.raises(TypeError):
            a - [1, 2]

    def test_mul_fixed_points(self):
        a = FixedPoint(0.5)
        b = FixedPoint(0.5)
        result = a * b
        assert abs(result.to_float() - 0.25) < 1e-7

    def test_mul_int(self):
        a = FixedPoint(0.25)
        result = a * 2
        assert abs(result.to_float() - 0.5) < 1e-7

    def test_mul_unsupported_type(self):
        a = FixedPoint(0.5)
        with pytest.raises(TypeError):
            a * 0.5

    def test_div_fixed_points(self):
        a = FixedPoint(0.5)
        b = FixedPoint(0.25)
        result = a / b
        assert abs(result.to_float() - 2.0) < 1e-5

    def test_div_int(self):
        a = FixedPoint(0.5)
        result = a / 2
        assert abs(result.to_float() - 0.25) < 1e-7

    def test_div_unsupported_type(self):
        a = FixedPoint(0.5)
        with pytest.raises(TypeError):
            a / "x"

    def test_lshift(self):
        a = FixedPoint(0.25)
        result = a << 1
        assert abs(result.to_float() - 0.5) < 1e-7

    def test_rshift(self):
        a = FixedPoint(0.5)
        result = a >> 1
        assert abs(result.to_float() - 0.25) < 1e-7

    def test_and_fixed_points(self):
        a = FixedPoint(0.5)
        b = FixedPoint(0.5)
        result = a & b
        assert isinstance(result, FixedPoint)

    def test_and_int(self):
        a = FixedPoint(0.5)
        result = a & 0xFF
        assert isinstance(result, FixedPoint)

    def test_and_unsupported_type(self):
        a = FixedPoint(0.5)
        with pytest.raises(TypeError):
            a & "x"

    def test_or_fixed_points(self):
        a = FixedPoint(0.5)
        b = FixedPoint(0.25)
        result = a | b
        assert isinstance(result, FixedPoint)

    def test_or_int(self):
        a = FixedPoint(0.5)
        result = a | 0x01
        assert isinstance(result, FixedPoint)

    def test_or_unsupported_type(self):
        a = FixedPoint(0.5)
        with pytest.raises(TypeError):
            a | 3.14


class TestFixedPointConversions:
    def test_to_int(self):
        fp = FixedPoint(3.7)
        assert fp.to_int() == 3

    def test_to_unsafe_int(self):
        fp = FixedPoint(0.5)
        assert fp.to_unsafe_int() == fp.value

    def test_to_float(self):
        fp = FixedPoint(0.375)
        assert abs(fp.to_float() - 0.375) < 1e-7

    def test_from_int(self):
        raw = 1 << 27  # represents 0.5 with 28 fractional bits
        fp = FixedPoint.from_int(raw)
        assert abs(fp.to_float() - 0.5) < 1e-7

    def test_from_int_custom_bits(self):
        raw = 1 << 15
        fp = FixedPoint.from_int(raw, fractional_bits=16, bit_width=32)
        assert abs(fp.to_float() - 0.5) < 1e-4

    def test_repr(self):
        fp = FixedPoint(0.5)
        r = repr(fp)
        assert "0.5" in r
