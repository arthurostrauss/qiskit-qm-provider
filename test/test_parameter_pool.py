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
        # The registry holds weak references to ParameterTables (only weak-referenceable
        # objects are stored). A registered table round-trips by its id.
        t = ParameterTable({"x": 0.0}, name="reg_obj")
        assert ParameterPool.get_obj(t._id) is t

    def test_get_obj_missing_raises(self):
        with pytest.raises(KeyError):
            ParameterPool.get_obj(99999)

    def test_reset_clears_all(self):
        ParameterTable({"x": 0.0}, name="o1")
        ParameterTable({"y": 0.0}, name="o2")
        assert len(ParameterPool.get_all_ids()) >= 2
        ParameterPool.reset()
        assert len(ParameterPool.get_all_ids()) == 0

    def test_registered_table_survives_gc_without_external_ref(self):
        # Regression: the pool keeps a strong companion to its weak registry so a table
        # whose only other reference is a transient local cannot be garbage-collected
        # mid-program and silently vanish from iteration (which would mis-classify its
        # parameters as standalone). No local reference is retained here on purpose.
        import gc

        ParameterTable({"x": 0.0}, name="kept_alive")
        gc.collect()
        assert "kept_alive" in {obj.name for obj in ParameterPool.get_all_objs()}
        ParameterPool.reset()
        gc.collect()
        assert "kept_alive" not in {obj.name for obj in ParameterPool.get_all_objs()}

    def test_rejected_registration_does_not_consume_id(self):
        # A duplicate-name registration must be rejected BEFORE the counter advances, so
        # the id sequence stays contiguous/deterministic.
        ParameterTable({"x": 0.0}, name="dup")

        class _Named:
            name = "dup"

        baseline = ParameterPool.get_id()  # plain counter bump, no object
        with pytest.raises(ValueError):
            ParameterPool.get_id(_Named())
        assert ParameterPool.get_id() == baseline + 1


class TestParameterPoolCollections:
    def test_get_all_ids(self):
        t1 = ParameterTable({"x": 0.0}, name="a")
        t2 = ParameterTable({"y": 0.0}, name="b")
        ids = ParameterPool.get_all_ids()
        assert t1._id in ids
        assert t2._id in ids

    def test_get_all_objs(self):
        t1 = ParameterTable({"x": 0.0}, name="a")
        t2 = ParameterTable({"y": 0.0}, name="b")
        objs = ParameterPool.get_all_objs()
        assert t1 in objs
        assert t2 in objs

    def test_get_all_dict(self):
        t1 = ParameterTable({"x": 0.0}, name="x")
        result = ParameterPool.get_all()
        assert isinstance(result, dict)
        assert result[t1._id] is t1


class TestParameterPoolNamedRegistration:
    def test_duplicate_parameter_table_name_raises(self):
        p1 = Parameter("a", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
        ParameterTable([p1], name="policy")
        p2 = Parameter("b", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
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
        field = tables["mini_packet"]
        # One-field structs are promoted to the field Parameter; Quarc handle lives on
        # the synthetic wrapper table (see ParameterPool module docstring).
        assert isinstance(field, Parameter)
        assert field.name == "x"
        wrapper = field.opnic_table
        assert wrapper is not None
        assert wrapper.name == "MiniPacket"
        assert wrapper._var is not None
        # Quarc INCOMING (into QUA) maps 1:1 to qiskit Direction.INCOMING.
        assert field.direction == Direction.INCOMING

    def test_from_quarc_module_struct_name_equals_field_name(self):
        """Regression: Quarc struct key can match the sole field name (snake_case)."""
        pytest.importorskip("quarc")
        from quarc import BaseModule, Direction as QuarcDirection, Scalar, Struct

        class Dup(BaseModule):
            def __init__(self) -> None:
                super().__init__()
                st = Struct(
                    struct_name="max_input_state",
                    max_input_state=Scalar[int],
                )
                self.add_struct(st, QuarcDirection.INCOMING)

        tables = ParameterPool.from_quarc_module(Dup())
        field = tables["max_input_state"]
        assert isinstance(field, Parameter)
        assert field.name == "max_input_state"
        assert field.opnic_table is not None
        assert field.opnic_table.name == "max_input_state"


class TestParameterPoolFromQuarcModuleDict:
    """Mode 2 — serialized state dict + live runtime (classical entrypoint)."""

    def test_from_quarc_module_dict_resolves_snake_case_runtime_endpoints(self):
        pytest.importorskip("quarc")

        class _PolicyEndpoint:
            def __init__(self):
                self.mu = [0.0, 0.0]
                self.sigma = [0.0, 0.0]
                self.sent = False

            def send(self):
                self.sent = True

            def recv(self):
                pass

        class _Runtime:
            policy_params = _PolicyEndpoint()

        state = {
            "_structs": {
                "PolicyParams": {
                    "struct": {
                        "mu": {"type": "float", "length": 2},
                        "sigma": {"type": "float", "length": 2},
                    },
                    "incoming_stream_spec": {"id": 7},
                    "outgoing_stream_spec": None,
                }
            }
        }
        wrappers = ParameterPool.from_quarc_module(state, opnic_runtime=_Runtime())
        assert "policy_params" in wrappers
        table = wrappers["policy_params"]
        assert isinstance(table, ParameterTable)
        assert table._var is _Runtime.policy_params
        assert table._var_is_quarc_handle is False
        assert table.incoming_stream_id == 7
        assert {p.name for p in table.parameters} == {"mu", "sigma"}

    def test_from_quarc_module_dict_accepts_already_snake_struct_keys(self):
        class _Endpoint:
            def send(self):
                pass

            def recv(self):
                pass

        class _Runtime:
            input_state_vars = _Endpoint()

        state = {
            "_structs": {
                "input_state_vars": {
                    "struct": {"n": {"type": "int", "length": 1}},
                    "incoming_stream_spec": {"id": 3},
                    "outgoing_stream_spec": None,
                }
            }
        }
        wrappers = ParameterPool.from_quarc_module(state, opnic_runtime=_Runtime())
        field = wrappers["input_state_vars"]
        assert isinstance(field, Parameter)
        assert field.opnic_table._var is _Runtime.input_state_vars
        assert field.var_is_quarc_handle is False

    def test_from_quarc_module_dict_reconstructs_non_opnic_with_consistent_keys(self):
        state = {
            "_structs": {},
            "parameter_specs": [
                {
                    "name": "NRepsVar",
                    "qua_type": "int",
                    "is_array": False,
                    "length": 0,
                    "input_type": "INPUT_STREAM",
                },
                {
                    "name": "IoFlag",
                    "qua_type": "bool",
                    "is_array": False,
                    "length": 0,
                    "input_type": "IO1",
                },
                {
                    "name": "legacy_snake",
                    "attr_name": "legacy_snake",
                    "qua_type": "fixed",
                    "is_array": False,
                    "length": 0,
                    "input_type": "IO2",
                },
            ],
        }

        class _EmptyRuntime:
            pass

        wrappers = ParameterPool.from_quarc_module(state, opnic_runtime=_EmptyRuntime())
        assert "n_reps_var" in wrappers
        assert wrappers["n_reps_var"].input_type == InputType.INPUT_STREAM
        assert wrappers["n_reps_var"].var_is_quarc_handle is False
        assert "io_flag" in wrappers
        assert wrappers["io_flag"].input_type == InputType.IO1
        assert "legacy_snake" in wrappers
        assert wrappers["legacy_snake"].input_type == InputType.IO2

    def test_mode1_reconstruction_marks_quarc_handle(self):
        pytest.importorskip("quarc")
        from quarc import BaseModule, Direction as QuarcDirection, Scalar, Struct

        class Mini(BaseModule):
            def __init__(self) -> None:
                super().__init__()
                st = Struct(struct_name="MiniPacket", x=Scalar[float])
                self.add_struct(st, QuarcDirection.INCOMING)

        wrappers = ParameterPool.from_quarc_module(Mini())
        field = wrappers["mini_packet"]
        assert field.opnic_table._var_is_quarc_handle is True
        assert field.var_is_quarc_handle is True
