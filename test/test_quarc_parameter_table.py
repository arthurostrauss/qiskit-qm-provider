# Copyright 2026 Arthur Strauss
"""Tests for the two Quarc pipelines (``from_quarc_module`` / ``to_quarc_module``) and
the standalone OPNIC parameter promotion machinery.

Quick mental map:

- **Pipeline 1 (module-first)**: ``ParameterPool.from_quarc_module(my_module)`` wraps an
  externally-built ``BaseModule``'s structs as ``ParameterTable`` instances and binds
  ``my_module`` as the pool's accumulator. Tables built *after* this point eagerly
  emit onto ``my_module``.
- **Pipeline 2 (parameters-first)**: the user declares ``Parameter`` / ``ParameterTable``
  objects with ``input_type=OPNIC`` and calls ``ParameterPool.to_quarc_module()`` to
  lazily allocate a default ``BaseModule`` and sweep every unemitted OPNIC table onto
  it.

In both cases solo OPNIC ``Parameter``\\ s are promoted to a synthetic single-field
``ParameterTable`` only on first ``Parameter.declare_variable()`` call (P2 model:
"promote on use") — ``to_quarc_module()`` itself does *not* promote pending solo
parameters.
"""

import pytest

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
# Existing iter / wrapper smoke tests
# ---------------------------------------------------------------------------


def test_iter_opnic_parameter_tables_order():
    p1 = Parameter("a", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    t1 = ParameterTable([p1], name="t1")
    p2 = Parameter("b", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    t2 = ParameterTable([p2], name="t2")
    assert [x._id for x in ParameterPool.iter_opnic_parameter_tables()] == sorted(
        [t1._id, t2._id]
    )


def test_from_quarc_module_policy_like_struct():
    pytest.importorskip("quarc")
    from quarc import Array, BaseModule, Direction as QuarcDirection, Struct

    class Pol(BaseModule):
        def __init__(self) -> None:
            super().__init__()
            st = Struct(
                struct_name="PolicyParams",
                mu=Array[float, 2],
                sigma=Array[float, 2],
            )
            self.add_struct(st, QuarcDirection.INCOMING)

    tables = ParameterPool.from_quarc_module(Pol())
    assert "policy_params" in tables
    pt = tables["policy_params"]
    assert {p.name for p in pt.parameters} == {"mu", "sigma"}
    assert pt.direction == Direction.OUTGOING


# ---------------------------------------------------------------------------
# Pipeline 2 — parameters-first
# ---------------------------------------------------------------------------


def test_pipeline2_table_is_pending_until_to_quarc_module():
    """Constructing an OPNIC ParameterTable before any module is bound leaves the table
    pending: ``_struct_type`` is built (cheap), but ``_is_emitted`` is False and
    ``_var`` is None until ``to_quarc_module()`` flushes it onto a fresh module."""
    pytest.importorskip("quarc")
    from quarc import BaseModule

    p_mu = Parameter(
        "mu", [0.0, 0.0], input_type=InputType.OPNIC, direction=Direction.OUTGOING
    )
    p_sigma = Parameter(
        "sigma", [0.1, 0.1], input_type=InputType.OPNIC, direction=Direction.OUTGOING
    )
    table = ParameterTable([p_mu, p_sigma], name="PolicyParams")

    # Pre-flush: struct type is built, but no module / no handle.
    assert table._struct_type is not None
    assert table._is_emitted is False
    assert table._var is None
    assert ParameterPool.has_quarc_module() is False

    module = ParameterPool.to_quarc_module()
    assert isinstance(module, BaseModule)
    assert "PolicyParams" in module._structs
    assert table._is_emitted is True
    assert table._var is module._struct_handles[0]

    # Idempotent — no extra structs added on a second call.
    module2 = ParameterPool.to_quarc_module()
    assert module2 is module
    assert len(module._struct_handles) == 1


def test_pipeline2_table_eagerly_emits_after_module_is_bound():
    """Once the pool has a module, OPNIC ParameterTables built afterwards eagerly emit
    onto it at construction time (no extra to_quarc_module() call needed)."""
    pytest.importorskip("quarc")

    module = ParameterPool.to_quarc_module()  # binds default BaseModule
    p = Parameter("x", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    t = ParameterTable([p], name="EagerTable")

    assert t._is_emitted is True
    assert t._var is not None
    assert "EagerTable" in module._structs
    assert t._var is module._struct_handles[-1]


def test_to_quarc_module_does_not_promote_pending_standalone_parameters():
    """to_quarc_module() is side-effect-free w.r.t. pending standalone OPNIC Parameters
    — only declare_variable() (or another transport-level call) promotes them."""
    pytest.importorskip("quarc")

    p = Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    assert p in ParameterPool._pending_standalone_opnic
    assert p.opnic_table is None

    module = ParameterPool.to_quarc_module()
    assert "theta" not in module._structs
    assert p in ParameterPool._pending_standalone_opnic
    assert p.opnic_table is None


def test_standalone_promote_via_require_table_creates_synthetic_and_locks():
    """Calling ``_require_standalone_opnic_table`` (used by declare_variable etc.)
    promotes the parameter to a synthetic single-field ParameterTable and locks it."""
    pytest.importorskip("quarc")

    p = Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    assert p.is_stand_alone
    assert p.opnic_table is None

    synthetic = p._require_standalone_opnic_table(context="declare_variable")

    # Synthetic table created, parameter locked into it.
    assert synthetic.name == "theta"
    assert synthetic._is_synthetic_standalone is True
    assert p.main_table is synthetic
    assert p.opnic_table is synthetic
    assert p not in ParameterPool._pending_standalone_opnic
    assert p.is_stand_alone  # standalone == True even after promotion (synthetic case)

    # Module was lazily created and the synthetic struct is in it.
    module = ParameterPool.quarc_module()
    assert "theta" in module._structs
    assert synthetic._var is not None

    # Re-attach to a different table fails.
    with pytest.raises(ValueError, match="standalone"):
        ParameterTable([p], name="other_table")


def test_pending_parameter_can_still_be_attached_to_a_regular_table():
    """A solo OPNIC Parameter that has not been declared yet is free to join a
    multi-field ParameterTable. After attachment, it is no longer pending and not
    treated as standalone."""
    pytest.importorskip("quarc")

    p = Parameter("phi", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    assert p in ParameterPool._pending_standalone_opnic

    table = ParameterTable([p], name="Holder")
    assert p.main_table is table
    assert p.opnic_table is None  # never promoted
    assert p not in ParameterPool._pending_standalone_opnic
    assert p.is_stand_alone is False

    module = ParameterPool.to_quarc_module()
    assert "Holder" in module._structs
    assert "phi" not in module._structs


# ---------------------------------------------------------------------------
# Pipeline 1 — module-first
# ---------------------------------------------------------------------------


def test_from_quarc_module_binds_pool_module_and_sweeps_pending_tables():
    """``from_quarc_module(my_module)`` should bind ``my_module`` as the pool's module
    AND emit any *previously* pending OPNIC ParameterTables onto it."""
    pytest.importorskip("quarc")
    from quarc import Array, BaseModule, Direction as QuarcDirection, Struct

    # User constructs a pending OPNIC table BEFORE calling from_quarc_module.
    p = Parameter(
        "z", [0.0, 0.0], input_type=InputType.OPNIC, direction=Direction.OUTGOING
    )
    pre = ParameterTable([p], name="Pre")
    assert pre._is_emitted is False

    # User builds a custom module separately.
    class M(BaseModule):
        def __init__(self) -> None:
            super().__init__()
            st = Struct(struct_name="ExistingStruct", x=Array[float, 1])
            self.add_struct(st, QuarcDirection.INCOMING)

    my_module = M()
    wrappers = ParameterPool.from_quarc_module(my_module)

    # The wrapper for ExistingStruct was produced.
    assert "existing_struct" in wrappers
    # ``Pre`` got swept onto my_module.
    assert pre._is_emitted is True
    assert "Pre" in my_module._structs
    assert pre._var is my_module._struct_handles[-1]
    # ``my_module`` is now the pool slot.
    assert ParameterPool.quarc_module() is my_module


def test_from_quarc_module_then_to_quarc_module_returns_same_module():
    pytest.importorskip("quarc")
    from quarc import BaseModule

    my_module = BaseModule()
    ParameterPool.from_quarc_module(my_module)
    assert ParameterPool.to_quarc_module() is my_module


def test_double_bind_raises():
    """Cannot call from_quarc_module after a module is already bound (e.g. by
    to_quarc_module)."""
    pytest.importorskip("quarc")
    from quarc import BaseModule

    ParameterPool.to_quarc_module()  # auto-allocates default BaseModule
    with pytest.raises(RuntimeError, match="already has a Quarc module"):
        ParameterPool.from_quarc_module(BaseModule())


def test_set_quarc_module_raises_on_double_bind():
    pytest.importorskip("quarc")
    from quarc import BaseModule

    ParameterPool.set_quarc_module(BaseModule())
    with pytest.raises(RuntimeError, match="already has a Quarc module"):
        ParameterPool.set_quarc_module(BaseModule())


# ---------------------------------------------------------------------------
# Parameter.__new__ — Option 1 (validating dedup, OPNIC-strict)
# ---------------------------------------------------------------------------


def test_dedup_opnic_collision_raises():
    """Re-constructing an OPNIC Parameter with the same name as an existing one (table-
    bound or pending) is rejected — OPNIC parameters are single-owner."""
    p1 = Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    assert p1 in ParameterPool._pending_standalone_opnic
    with pytest.raises(ValueError, match="OPNIC parameters cannot be re-declared"):
        Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)


def test_dedup_opnic_collision_with_table_member_raises():
    p1 = Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    ParameterTable([p1], name="Holder")
    with pytest.raises(ValueError, match="OPNIC parameters cannot be re-declared"):
        Parameter("theta", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)


def test_dedup_non_opnic_compatible_returns_existing():
    """Re-constructing a non-OPNIC Parameter with the same name and matching args
    returns the existing instance."""
    from qm.qua import fixed

    p1 = Parameter("alpha", 0.5, qua_type=fixed)
    table = ParameterTable([p1], name="Tab")
    p2 = Parameter("alpha", 0.5, qua_type=fixed)
    assert p2 is p1


def test_dedup_non_opnic_incompatible_raises():
    """Re-constructing a non-OPNIC Parameter with conflicting attributes raises."""
    p1 = Parameter("alpha", 0.5)  # qua_type inferred as fixed
    ParameterTable([p1], name="Tab")
    with pytest.raises(ValueError, match="different attributes"):
        Parameter("alpha", 0.5, qua_type=int)


def test_dedup_non_opnic_compatible_omitted_args_returns_existing():
    """``Parameter('foo')`` lookup returns the existing instance when no conflicting
    args are passed, *provided* the existing instance is discoverable through a
    registered ParameterTable. Bare standalone non-OPNIC parameters aren't tracked in
    the pool, matching the historical pre-Option-1 behavior."""
    from qm.qua import fixed

    p1 = Parameter("alpha", 0.5, qua_type=fixed)
    ParameterTable([p1], name="Tab")  # makes p1 discoverable via the pool registry.
    p2 = Parameter("alpha")  # all args except name omitted — compatible.
    assert p2 is p1


# ---------------------------------------------------------------------------
# Field-level OPNIC operations on table-attached parameter must raise
# ---------------------------------------------------------------------------


def test_table_attached_opnic_parameter_field_methods_raise_table_managed():
    p = Parameter(
        "theta", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING
    )
    ParameterTable([p], name="packet")

    with pytest.raises(RuntimeError, match="table-managed"):
        p.declare_variable()
    with pytest.raises(RuntimeError, match="table-managed"):
        p.push_to_opx(0.5)
    with pytest.raises(RuntimeError, match="table-managed"):
        p.fetch_from_opx()
    with pytest.raises(RuntimeError, match="table-managed"):
        p.load_input_value()
    with pytest.raises(RuntimeError, match="table-managed"):
        p.stream_back()


# ---------------------------------------------------------------------------
# iter_standalone_opnic_parameters — union of pending + promoted
# ---------------------------------------------------------------------------


def test_iter_standalone_opnic_parameters_unions_pending_and_promoted():
    pytest.importorskip("quarc")

    pending = Parameter(
        "p_pending", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING
    )
    promoted = Parameter(
        "p_promoted", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING
    )
    promoted._require_standalone_opnic_table(context="declare_variable")

    members = ParameterPool.iter_standalone_opnic_parameters()
    assert pending in members
    assert promoted in members


# ---------------------------------------------------------------------------
# Annotation inference round-trip
# ---------------------------------------------------------------------------


def test_quarc_annotation_inference_matches_qua_types():
    pytest.importorskip("quarc")
    from quarc import Array, Scalar
    from typing import get_args, get_origin

    from qiskit_qm_provider.parameter_table.quarc_emit import quarc_annotation_for

    p_int = Parameter("i", 1, qua_type=int)
    p_float = Parameter("f", 0.5)  # inferred as fixed
    p_bool = Parameter("b", False)
    p_arr = Parameter("a", [0.0, 0.1, 0.2])

    a_int = quarc_annotation_for(p_int)
    a_float = quarc_annotation_for(p_float)
    a_bool = quarc_annotation_for(p_bool)
    a_arr = quarc_annotation_for(p_arr)

    assert get_origin(a_int) is Scalar and get_args(a_int) == (int,)
    assert get_origin(a_float) is Scalar and get_args(a_float) == (float,)
    assert get_origin(a_bool) is Scalar and get_args(a_bool) == (bool,)
    assert get_origin(a_arr) is Array and get_args(a_arr) == (float, 3)
