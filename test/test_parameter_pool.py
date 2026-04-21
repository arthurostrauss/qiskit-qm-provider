"""Tests for ParameterPool."""

import pytest

from qiskit_qm_provider import (
    Direction,
    InputType,
    Parameter,
    ParameterPool,
    ParameterTable,
)


@pytest.fixture(autouse=True)
def reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


class TestParameterPoolIdManagement:
    def test_get_id_increments(self):
        id1 = ParameterPool.get_id()
        id2 = ParameterPool.get_id()
        assert id2 == id1 + 1

    def test_get_id_with_object(self):
        obj = {"test": True}
        id_ = ParameterPool.get_id(obj)
        assert ParameterPool.get_obj(id_) is obj

    def test_get_obj_missing_raises(self):
        with pytest.raises(KeyError):
            ParameterPool.get_obj(99999)

    def test_reset_clears_all(self):
        ParameterPool.get_id("obj1")
        ParameterPool.get_id("obj2")
        assert len(ParameterPool.get_all_ids()) >= 2
        ParameterPool.reset()
        assert len(ParameterPool.get_all_ids()) == 0


class TestParameterPoolCollections:
    def test_get_all_ids(self):
        id1 = ParameterPool.get_id("a")
        id2 = ParameterPool.get_id("b")
        ids = ParameterPool.get_all_ids()
        assert id1 in ids
        assert id2 in ids

    def test_get_all_objs(self):
        ParameterPool.get_id("a")
        ParameterPool.get_id("b")
        objs = ParameterPool.get_all_objs()
        assert "a" in objs
        assert "b" in objs

    def test_get_all_dict(self):
        id1 = ParameterPool.get_id("x")
        result = ParameterPool.get_all()
        assert isinstance(result, dict)
        assert result[id1] == "x"


class TestParameterPoolNamedRegistration:
    def test_duplicate_parameter_table_name_raises(self):
        p1 = Parameter(
            "a", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING
        )
        ParameterTable([p1], name="policy")
        p2 = Parameter(
            "b", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING
        )
        with pytest.raises(ValueError, match="Duplicate pool registration name"):
            ParameterTable([p2], name="policy")


class TestParameterPoolFromQuarcModule:
    def test_from_quarc_module_smoke(self):
        pytest.importorskip("quarc")
        from quarc import BaseModule, Direction as QuarcDirection, Scalar, Struct

        class Mini(BaseModule):
            def __init__(self) -> None:
                super().__init__()
                st = Struct(struct_name="MiniPacket", x=Scalar[float])
                self.add_struct(st, QuarcDirection.INCOMING)

        m = Mini()
        tables = ParameterPool.from_quarc_module(m)
        assert "mini_packet" in tables
        t = tables["mini_packet"]
        assert t.name == "MiniPacket"
        assert t._var is not None
        assert t.direction == Direction.OUTGOING
