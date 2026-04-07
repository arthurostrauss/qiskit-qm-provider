# Copyright 2026 Arthur Strauss
"""Tests for Quarc stream id assignment and Quarc hybrid alignment."""

import pytest

from qiskit_qm_provider import (
    Direction,
    InputType,
    Parameter,
    ParameterPool,
    ParameterTable,
    default_quarc_struct_name,
)

pytest.importorskip("quarc")


@pytest.fixture(autouse=True)
def _reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


def test_attach_quarc_stream_specs_rejects_non_opnic():
    p = Parameter("x", 0.0)
    t = ParameterTable([p], name="t")
    with pytest.raises(ValueError, match="OPNIC"):
        t._attach_quarc_stream_specs(incoming_id=1, outgoing_id=2)


def test_opnic_qua_edges_require_prepare():
    p = Parameter("x", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    t = ParameterTable([p], name="policy")
    with pytest.raises(RuntimeError, match="prepare_opnic_quarc_hybrid_packets"):
        t._stream_id_for_qua_incoming_edge()


def test_prepare_opnic_assigns_via_quarc_module():
    p1 = Parameter("a", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    t1 = ParameterTable([p1], name="t1")
    p2 = Parameter("b", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    t2 = ParameterTable([p2], name="t2")
    ParameterPool.prepare_opnic_quarc_hybrid_packets()
    assert t1._quarc_outgoing_stream_id == 1
    assert t1._quarc_incoming_stream_id is None
    assert t2._quarc_incoming_stream_id == 1
    assert t2._quarc_outgoing_stream_id is None


def test_iter_opnic_parameter_tables_order():
    p1 = Parameter("a", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    t1 = ParameterTable([p1], name="t1")
    p2 = Parameter("b", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    t2 = ParameterTable([p2], name="t2")
    assert [x._id for x in ParameterPool.iter_opnic_parameter_tables()] == sorted([t1._id, t2._id])


def test_default_quarc_struct_name_standalone_parameter():
    p = Parameter("solo", 1.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    ParameterPool.prepare_opnic_quarc_hybrid_packets()
    assert default_quarc_struct_name(p) == f"Packet_solo_{p.stream_id}"


def test_prepare_standalone_both():
    p = Parameter("solo", 1.0, input_type=InputType.OPNIC, direction=Direction.BOTH)
    ParameterPool.prepare_opnic_quarc_hybrid_packets()
    assert p._quarc_incoming_stream_id == 1
    assert p._quarc_outgoing_stream_id == 1


def test_incoming_table_qua_outgoing_edge_uses_quarc_incoming_spec():
    p = Parameter("y", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)
    t = ParameterTable([p], name="reward")
    ParameterPool.prepare_opnic_quarc_hybrid_packets()
    assert t._stream_id_for_qua_outgoing_edge() == 1
    assert t._quarc_incoming_stream_id == 1


def test_outgoing_table_qua_incoming_edge_uses_quarc_outgoing_spec():
    p = Parameter("x", 0.0, input_type=InputType.OPNIC, direction=Direction.OUTGOING)
    t = ParameterTable([p], name="policy")
    ParameterPool.prepare_opnic_quarc_hybrid_packets()
    assert t._stream_id_for_qua_incoming_edge() == 1
    assert t._quarc_outgoing_stream_id == 1
