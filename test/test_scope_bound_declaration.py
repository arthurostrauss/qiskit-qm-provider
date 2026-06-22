# Copyright 2026 Arthur Strauss
"""Scope-bound declaration validity (Phase 2 isolation slice).

``Parameter.is_declared`` is tied to the QUA *program* scope it was declared in, so a
fresh ``with program():`` auto-invalidates a prior declaration — the same object can be
re-declared across programs without a manual ``reset()`` and without leaking stale QUA
state. ``ParameterPool.reset()`` also resets Quarc's process-global stream-id counters.
"""

import warnings

import pytest
from qm.qua import program

from qiskit_qm_provider import Parameter, ParameterPool, ParameterTable


def test_is_declared_is_scope_bound():
    p = Parameter("theta", 0.0)
    t = ParameterTable([p], name="T")

    assert p.is_declared is False  # never declared
    with program():
        t.declare()
        assert p.is_declared is True
    # Outside any program scope the declaration is no longer valid.
    assert p.is_declared is False


def test_redeclare_in_new_program_is_clean_and_silent():
    p = Parameter("theta", 0.0)
    t = ParameterTable([p], name="T")

    with program():
        t.declare()

    # A fresh program auto-invalidates the prior declaration; re-declaring must not raise
    # "already declared" and must not warn about a stale stream.
    with program():
        assert p.is_declared is False
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any stale-stream warning becomes an error
            t.declare()
        assert p.is_declared is True


def test_reset_resets_quarc_stream_id_counters():
    quarc_streams = pytest.importorskip("quarc.dsl.streams")
    quarc_streams._incoming_ids.next()
    quarc_streams._outgoing_ids.next()
    ParameterPool.reset()
    assert quarc_streams._incoming_ids.current == 0
    assert quarc_streams._outgoing_ids.current == 0
