"""Tests for FixedPoint arithmetic."""

import warnings

import pytest
from qiskit_qm_provider.fixed_point import FixedPoint, FixedPointApproximationWarning, FixedPointOverflowWarning


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

    def test_out_of_range_constant_wraps_and_warns(self):
        with pytest.warns(FixedPointOverflowWarning):
            assert FixedPoint(9.0).to_float() == -7.0
        with pytest.warns(FixedPointOverflowWarning):
            assert FixedPoint(-9.0).to_float() == 7.0

    def test_range_edges_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FixedPointOverflowWarning)
            assert FixedPoint(-8.0).value == -(1 << 31)
            assert FixedPoint.from_int((1 << 31) - 1).value == (1 << 31) - 1


class TestFixedPointOverflow:
    """Wrap-around semantics, checked against QUA fixed arithmetic on hardware (IQCC 'arbel')."""

    @pytest.mark.parametrize(
        "op, x, y, expected",
        [
            ("add", 7.0, 3.0, -6.0),
            ("add", -7.0, -3.0, 6.0),
            ("sub", 7.0, -3.0, -6.0),
            ("mul", 3.0, 3.0, -7.0),
            ("mul", -3.0, 3.0, 7.0),
        ],
    )
    def test_hardware_overflow_cases_wrap_and_warn(self, op, x, y, expected):
        fx, fy = FixedPoint(x), FixedPoint(y)
        with pytest.warns(FixedPointOverflowWarning, match="wraps around"):
            result = {"add": fx + fy, "sub": fx - fy, "mul": fx * fy}[op]
        assert result.to_float() == expected

    @pytest.mark.parametrize("op, x, y, expected", [("add", 3.0, 2.0, 5.0), ("mul", 2.0, 3.0, 6.0)])
    def test_in_range_does_not_warn(self, op, x, y, expected):
        fx, fy = FixedPoint(x), FixedPoint(y)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FixedPointOverflowWarning)
            result = fx + fy if op == "add" else fx * fy
        assert result.to_float() == expected

    def test_shift_left_wraps(self):
        with pytest.warns(FixedPointOverflowWarning):
            assert (FixedPoint(5.0) << 1).to_float() == -6.0

    def test_warning_can_be_escalated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FixedPointOverflowWarning)
            with pytest.raises(FixedPointOverflowWarning):
                FixedPoint(7.0) + FixedPoint(3.0)

    def test_warning_points_at_caller(self):
        with pytest.warns(FixedPointOverflowWarning) as record:
            FixedPoint(7.0) + FixedPoint(3.0)
        assert record[0].filename == __file__


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

    @pytest.mark.parametrize("value", [0.1, -0.2, 0.3, 0.05, 0.9999999, -1.3333333])
    def test_rounds_to_nearest_like_qua(self, value):
        # QUA's declare(fixed, value=...) rounds to nearest (checked on hardware).
        assert FixedPoint(value).value == round(value * (1 << 28))

    def test_float_builtin(self):
        fp = FixedPoint(-1.625)
        assert float(fp) == fp.to_float()

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


class TestFixedPointDivisionWarning:
    def test_division_warns_it_is_approximate_on_hardware(self):
        with pytest.warns(FixedPointApproximationWarning, match="approximate on the controller") as record:
            result = FixedPoint(1.0) / FixedPoint(4.0)
        assert result.to_float() == 0.25
        assert record[0].filename == __file__
