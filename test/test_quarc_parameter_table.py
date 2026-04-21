# Copyright 2026 Arthur Strauss
"""Tests for :meth:`ParameterPool.from_quarc_module` and OPNIC table iteration."""

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
    from quarc import Array, BaseModule, Direction as QuarcDirection, Scalar, Struct

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


def test_opnic_parameter_field_methods_are_table_only():
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
