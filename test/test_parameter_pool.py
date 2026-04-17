"""Tests for ParameterPool."""

import pytest
from qiskit_qm_provider.parameter_table.parameter_pool import ParameterPool


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


class TestParameterPoolDunderMethods:
    def test_getitem(self):
        pool = ParameterPool()
        id_ = ParameterPool.get_id("item")
        assert pool[id_] == "item"

    def test_setitem_raises_on_duplicate(self):
        pool = ParameterPool()
        id_ = ParameterPool.get_id("item")
        with pytest.raises(ValueError, match="already exists"):
            pool[id_] = "another"

    def test_delitem(self):
        pool = ParameterPool()
        id_ = ParameterPool.get_id("item")
        del pool[id_]
        assert id_ not in pool

    def test_delitem_missing_raises(self):
        pool = ParameterPool()
        with pytest.raises(KeyError):
            del pool[99999]

    def test_contains(self):
        pool = ParameterPool()
        id_ = ParameterPool.get_id("item")
        assert id_ in pool
        assert 99999 not in pool

    def test_len(self):
        pool = ParameterPool()
        ParameterPool.get_id("a")
        ParameterPool.get_id("b")
        assert len(pool) >= 2

    def test_iter(self):
        pool = ParameterPool()
        ParameterPool.get_id("a")
        ParameterPool.get_id("b")
        items = list(pool)
        assert "a" in items
        assert "b" in items

    def test_str(self):
        pool = ParameterPool()
        ParameterPool.get_id("x")
        s = str(pool)
        assert "x" in s

    def test_repr(self):
        pool = ParameterPool()
        r = repr(pool)
        assert "ParameterPool" in r


class TestParameterPoolFlags:
    def test_patched_default_false(self):
        assert ParameterPool.patched() is False

    def test_configured_default_false(self):
        assert ParameterPool.configured() is False
