"""Tests for ParameterPool."""

import pytest

from qiskit_qm_provider import Direction, InputType, Parameter, ParameterPool, ParameterTable
from qiskit_qm_provider.parameter_table.quarc_naming import default_quarc_struct_name


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


class TestOpnicTransportModule:
    def test_set_opnic_transport_module_override(self):
        class FakeTransport:
            pass

        ParameterPool.set_opnic_transport_module(FakeTransport)
        assert ParameterPool.get_opnic_transport_module() is FakeTransport


class TestParameterPoolNamedRegistration:
    def test_duplicate_parameter_table_name_raises(self):
        p1 = Parameter("a", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
        ParameterTable([p1], name="policy")
        p2 = Parameter("b", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
        with pytest.raises(ValueError, match="Duplicate pool registration name"):
            ParameterTable([p2], name="policy")


class TestParameterPoolRebind:
    def test_rebind_parameter_table_id_moves_registry_entry(self):
        mu = Parameter("mu", [0.0], input_type=InputType.OPNIC, direction=Direction.OUTGOING)
        sigma = Parameter("sigma", [1.0], input_type=InputType.OPNIC, direction=Direction.OUTGOING)
        t = ParameterTable([mu, sigma], name="policy")
        old_id = t._id
        ParameterPool.rebind_parameter_table_id(t, 42)
        assert t._id == 42
        assert ParameterPool.get_obj(42) is t
        assert old_id not in ParameterPool.get_all_ids()

    def test_rebind_standalone_opnic_parameter_stream_id(self):
        p = Parameter("solo_rebind_test", [0.0], input_type=InputType.OPNIC, direction=Direction.INCOMING)
        _ = p.stream_id
        ParameterPool.rebind_standalone_opnic_parameter_stream_id(p, 77)
        assert p._stream_id == 77
        assert ParameterPool.get_obj(77) is p


class TestParameterPoolQuarcSpecsAttach:
    def test_attach_opnic_streams_from_specs_dict_smoke(self):
        pytest.importorskip("quarc")
        mu = Parameter("mu", [0.0], input_type=InputType.OPNIC, direction=Direction.OUTGOING)
        sigma = Parameter("sigma", [1.0], input_type=InputType.OPNIC, direction=Direction.OUTGOING)
        tab = ParameterTable([mu, sigma], name="policy")
        tid = tab._id
        specs = {
            "version": 1,
            "structs": [
                {
                    "struct_name": default_quarc_struct_name(tab),
                    "direction": "OUTGOING",
                    "fields": [
                        {"name": "mu", "dtype": "float", "length": 1},
                        {"name": "sigma", "dtype": "float", "length": 1},
                    ],
                    "rl_qoc_binding": "policy",
                    "rl_qoc_pool_table_id": tid,
                }
            ],
        }
        ParameterPool.attach_opnic_streams_from_specs_dict(specs)
