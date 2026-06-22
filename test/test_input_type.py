"""Tests for InputType and Direction enums."""

import pytest
from qiskit_qm_provider.parameter_table.input_type import InputType, Direction


class TestInputType:
    def test_members(self):
        assert InputType.OPNIC.value == "OPNIC"
        assert InputType.INPUT_STREAM.value == "INPUT_STREAM"
        assert InputType.IO1.value == "IO1"
        assert InputType.IO2.value == "IO2"

    def test_str(self):
        assert str(InputType.OPNIC) == "OPNIC"
        assert str(InputType.INPUT_STREAM) == "INPUT_STREAM"

    def test_from_string_valid(self):
        assert InputType.from_string("OPNIC") == InputType.OPNIC
        assert InputType.from_string("INPUT_STREAM") == InputType.INPUT_STREAM
        assert InputType.from_string("IO1") == InputType.IO1
        assert InputType.from_string("IO2") == InputType.IO2

    def test_from_string_none(self):
        assert InputType.from_string(None) is None

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid input type"):
            InputType.from_string("INVALID")

    def test_construct_from_value(self):
        assert InputType("OPNIC") == InputType.OPNIC
        assert InputType("IO1") == InputType.IO1

    def test_construct_from_invalid_value(self):
        with pytest.raises(ValueError):
            InputType("NONEXISTENT")

    def test_all_members_count(self):
        assert len(InputType) == 4

    def test_equality(self):
        assert InputType.OPNIC == InputType.OPNIC
        assert InputType.IO1 != InputType.IO2


class TestDirection:
    def test_members(self):
        assert Direction.INCOMING.value == "INCOMING"
        assert Direction.OUTGOING.value == "OUTGOING"
        assert Direction.BOTH.value == "BOTH"

    def test_str(self):
        assert str(Direction.INCOMING) == "INCOMING"
        assert str(Direction.OUTGOING) == "OUTGOING"
        assert str(Direction.BOTH) == "BOTH"

    def test_from_string_valid(self):
        assert Direction.from_string("INCOMING") == Direction.INCOMING
        assert Direction.from_string("OUTGOING") == Direction.OUTGOING
        assert Direction.from_string("BOTH") == Direction.BOTH

    def test_from_string_none(self):
        assert Direction.from_string(None) is None

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid direction"):
            Direction.from_string("INVALID")

    def test_all_members_count(self):
        assert len(Direction) == 3

    def test_construct_from_value(self):
        assert Direction("INCOMING") == Direction.INCOMING

    def test_equality(self):
        assert Direction.INCOMING != Direction.OUTGOING
