# Copyright 2026 Arthur Strauss
"""Tests for the two Quarc construction flows and the OPNIC emission model.

Model (post-redesign):

- **Flow A (generation / parameters-first)**: the user builds ``Parameter`` /
  ``ParameterTable`` objects with ``input_type=OPNIC``. The Quarc struct *type* is built
  cheaply at construction, but emission (``add_struct`` + stream-id minting) is **deferred
  to ``declare()`` inside a ``with program():``** — the single commit point. A lone OPNIC
  ``Parameter`` is promoted to a synthetic 1-field ``ParameterTable`` at its first
  ``declare()``. OPNIC is single-program-per-process: declaring under a second distinct
  program scope raises.
- **Flow B (reconstruction / module-first)**: ``ParameterPool.from_quarc_module(module)``
  wraps an externally-built module's pre-emitted structs as ``ParameterTable`` instances
  (1-field structs surface as a ``Parameter`` with ``.opnic_table``).

Parameters are no longer globally name-deduplicated: ``Parameter("x")`` twice yields two
distinct objects. Only *table* names must be unique (they are Quarc struct keys).
"""

import pytest
from qm.qua import program

from qiskit_qm_provider import (
    Direction,
    InputType,
    Parameter,
    ParameterPool,
    ParameterTable,
)


@pytest.fixture(autouse=True)
def _reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


# ---------------------------------------------------------------------------
# Iteration / Flow-B wrapper smoke tests
# ---------------------------------------------------------------------------


def test_iter_opnic_parameter_tables_order():
    p1 = Parameter("a", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    t1 = ParameterTable([p1], name="t1")
    p2 = Parameter("b", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    t2 = ParameterTable([p2], name="t2")
    assert [x._id for x in ParameterPool.iter_opnic_parameter_tables()] == sorted([t1._id, t2._id])


def test_from_quarc_module_policy_like_struct():
    pytest.importorskip("quarc")
    from quarc import Array, BaseModule, Direction as QuarcDirection, Struct

    class Pol(BaseModule):
        def __init__(self) -> None:
            super().__init__()
            st = Struct(struct_name="PolicyParams", mu=Array[float, 2], sigma=Array[float, 2])
            self.add_struct(st, QuarcDirection.INCOMING)

    tables = ParameterPool.from_quarc_module(Pol())
    assert "policy_params" in tables
    pt = tables["policy_params"]
    assert {p.name for p in pt.parameters} == {"mu", "sigma"}
    # Quarc INCOMING (into QUA) now maps 1:1 to qiskit Direction.INCOMING.
    assert pt.direction == Direction.INCOMING


# ---------------------------------------------------------------------------
# Flow A — emission deferred to declare()
# ---------------------------------------------------------------------------


def test_opnic_table_not_emitted_until_declare():
    """The Quarc struct type is built at construction, but no module/handle/stream id is
    created until ``declare()`` runs inside a program — even after a module is bound."""
    pytest.importorskip("quarc")
    from qiskit_qm_provider import QiskitQMModule

    p_mu = Parameter("mu", [0.0, 0.0], input_type=InputType.OPNIC, direction=Direction.INCOMING)
    p_sigma = Parameter("sigma", [0.1, 0.1], input_type=InputType.OPNIC, direction=Direction.INCOMING)
    table = ParameterTable([p_mu, p_sigma], name="PolicyParams")

    assert table._struct_type is None  # struct type is built at declare(), not construction
    assert table._is_emitted is False
    assert table._var is None

    # Binding a module does NOT emit (no eager sweep anymore).
    module = QiskitQMModule()
    assert "PolicyParams" not in module._structs
    assert table._is_emitted is False

    # Emission happens at declare() inside the program.
    with program():
        table.declare()
    assert table._is_emitted is True
    assert table._var is module._struct_handles[-1]
    assert "PolicyParams" in module._structs


def test_struct_type_reflects_fields_added_before_declare():
    """The Quarc struct type is built from the FINAL field set at declare(): parameters
    added after construction (but before declare) must appear in the emitted struct."""
    pytest.importorskip("quarc")

    t = ParameterTable([Parameter("a", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)], name="Dyn")
    # Add a field after construction — allowed while not yet emitted.
    t.add_parameters(Parameter("b", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING))
    assert t._struct_type is None  # nothing committed yet

    with program():
        t.declare()

    from typing import get_type_hints

    fields = set(get_type_hints(t.struct_type).keys())
    assert fields == {"a", "b"}  # the dynamically-added field 'b' is present


def test_opnic_table_emits_onto_default_module_at_declare():
    """With no module pre-bound, declare() lazily creates the default QiskitQMModule and
    emits onto it."""
    pytest.importorskip("quarc")

    p = Parameter("x", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    t = ParameterTable([p], name="EagerTable")
    assert t._is_emitted is False

    with program():
        t.declare()

    module = ParameterPool.quarc_module()
    assert t._is_emitted is True
    assert "EagerTable" in module._structs


def test_standalone_parameter_promotes_and_emits_at_declare():
    """A lone OPNIC Parameter is pending until first declare(), then promoted to a
    synthetic 1-field table that emits its struct."""
    pytest.importorskip("quarc")

    p = Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    assert p in ParameterPool._pending_standalone_opnic
    assert p.opnic_table is None

    with program():
        p.declare()

    assert p.opnic_table is not None
    assert p.opnic_table.name == "theta_packet"
    assert p not in ParameterPool._pending_standalone_opnic
    module = ParameterPool.quarc_module()
    assert "theta_packet" in module._structs


def test_single_program_guard_raises_on_second_program():
    """OPNIC is one-program-per-process: declaring OPNIC tables under a second distinct
    program scope raises (unless ParameterPool.reset() is called first)."""
    pytest.importorskip("quarc")

    p1 = Parameter("a", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    t1 = ParameterTable([p1], name="First")
    with program():
        t1.declare()

    p2 = Parameter("b", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    t2 = ParameterTable([p2], name="Second")
    with program():
        with pytest.raises(RuntimeError, match="second, distinct QUA program scope"):
            t2.declare()


def test_pending_parameter_can_still_join_a_regular_table():
    """A solo OPNIC Parameter that has not been declared yet is free to join a multi-field
    table; afterwards it is no longer pending nor standalone."""
    pytest.importorskip("quarc")

    p = Parameter("phi", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    assert p in ParameterPool._pending_standalone_opnic

    table = ParameterTable([p], name="Holder")
    assert p.main_table is table
    assert p.opnic_table is None  # never promoted
    assert p not in ParameterPool._pending_standalone_opnic
    assert p.is_stand_alone is False


# ---------------------------------------------------------------------------
# Flow B — module-first
# ---------------------------------------------------------------------------


def test_from_quarc_module_binds_pool_module_without_sweeping_plain_base_module():
    pytest.importorskip("quarc")
    from quarc import Array, BaseModule, Direction as QuarcDirection, Struct

    p = Parameter("z", [0.0, 0.0], input_type=InputType.OPNIC, direction=Direction.INCOMING)
    pre = ParameterTable([p], name="Pre")
    assert pre._is_emitted is False

    class M(BaseModule):
        def __init__(self) -> None:
            super().__init__()
            st = Struct(struct_name="ExistingStruct", x=Array[float, 1])
            self.add_struct(st, QuarcDirection.INCOMING)

    my_module = M()
    wrappers = ParameterPool.from_quarc_module(my_module)

    assert "existing_struct" in wrappers
    assert pre._is_emitted is False  # never declared -> never emitted
    assert "Pre" not in my_module._structs
    assert ParameterPool.quarc_module() is my_module


def test_from_quarc_module_then_to_quarc_module_returns_same_module():
    pytest.importorskip("quarc")
    from quarc import BaseModule

    my_module = BaseModule()
    ParameterPool.from_quarc_module(my_module)
    assert ParameterPool.to_quarc_module() is my_module


def test_double_bind_raises():
    pytest.importorskip("quarc")
    from quarc import BaseModule

    ParameterPool.to_quarc_module()  # auto-allocates default QiskitQMModule
    with pytest.raises(RuntimeError, match="already has a Quarc module"):
        ParameterPool.from_quarc_module(BaseModule())


def test_set_quarc_module_raises_on_double_bind():
    pytest.importorskip("quarc")
    from quarc import BaseModule

    ParameterPool.set_quarc_module(BaseModule())
    with pytest.raises(RuntimeError, match="already has a Quarc module"):
        ParameterPool.set_quarc_module(BaseModule())


# ---------------------------------------------------------------------------
# No global Parameter name-dedup (only table names are unique)
# ---------------------------------------------------------------------------


def test_parameters_with_same_name_are_distinct_objects():
    """Re-constructing a Parameter with the same name yields a NEW object (the old global
    name-dedup is gone)."""
    from qm.qua import fixed

    p1 = Parameter("alpha", 0.5, qua_type=fixed)
    ParameterTable([p1], name="Tab")
    p2 = Parameter("alpha", 0.5, qua_type=fixed)
    assert p2 is not p1

    # OPNIC params named the same are also distinct now.
    o1 = Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    o2 = Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    assert o1 is not o2


def test_duplicate_table_name_still_raises():
    """Table names remain unique (they become Quarc struct keys)."""
    p1 = Parameter("a", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    ParameterTable([p1], name="policy")
    p2 = Parameter("b", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    with pytest.raises(ValueError, match="Duplicate pool registration name"):
        ParameterTable([p2], name="policy")


# ---------------------------------------------------------------------------
# Field-level OPNIC operations on a table-attached parameter must raise
# ---------------------------------------------------------------------------


def test_table_attached_opnic_parameter_field_methods_raise_table_managed():
    p = Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    ParameterTable([p], name="packet")

    with pytest.raises(RuntimeError, match="table-managed"):
        p.declare_variable()
    with pytest.raises(RuntimeError, match="table-managed"):
        p.push_to_opx(0.5)
    with pytest.raises(RuntimeError, match="table-managed"):
        p.fetch_from_opx()


# ---------------------------------------------------------------------------
# Annotation inference round-trip
# ---------------------------------------------------------------------------


def test_quarc_annotation_inference_matches_qua_types():
    pytest.importorskip("quarc")
    from quarc import Array, Scalar
    from typing import get_args, get_origin

    from qiskit_qm_provider.parameter_table.quarc_emit import quarc_annotation_for

    assert get_origin(quarc_annotation_for(Parameter("i", 1, qua_type=int))) is Scalar
    assert get_args(quarc_annotation_for(Parameter("i2", 1, qua_type=int))) == (int,)
    assert get_args(quarc_annotation_for(Parameter("f", 0.5))) == (float,)
    assert get_args(quarc_annotation_for(Parameter("b", False))) == (bool,)
    a_arr = quarc_annotation_for(Parameter("a", [0.0, 0.1, 0.2]))
    assert get_origin(a_arr) is Array and get_args(a_arr) == (float, 3)
