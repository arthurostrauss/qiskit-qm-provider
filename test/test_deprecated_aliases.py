# Copyright 2026 Arthur Strauss
"""Tests that deprecated method aliases on Parameter and ParameterTable emit
:class:`DeprecationWarning` and still resolve to the canonical implementation.

The descriptor emits the warning on *attribute access* (before the call), so
pytest.warns captures it even when the underlying QUA call raises (no program
context outside a real QUA program). All QUA-touching calls are wrapped in
try/except inside the warns block for this reason.

"No-warning" tests assert at the *class attribute level* that canonical names
are plain methods, not ``_DeprecatedAlias`` descriptors.
"""

import pytest

from qiskit_qm_provider import Parameter, ParameterTable, ParameterPool, InputType
from qiskit_qm_provider.parameter_table._deprecation import _DeprecatedAlias


@pytest.fixture(autouse=True)
def _reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input_stream_param(name: str = "p") -> Parameter:
    return Parameter(name, value=0.0, qua_type="fixed", input_type=InputType.INPUT_STREAM)


def _make_input_stream_table(names=("a", "b")) -> ParameterTable:
    params = [Parameter(n, value=0.0, qua_type="fixed", input_type=InputType.INPUT_STREAM) for n in names]
    return ParameterTable(params, name="test_table")


def _assert_deprecation(w, *, old: str, new: str, version: str = "1.2"):
    messages = [str(warning.message) for warning in w]
    assert any(
        old in msg and new in msg and f"v{version}" in msg
        for msg in messages
    ), f"Expected deprecation for {old!r} → {new!r} in v{version}, got: {messages}"


def _call_safely(fn):
    """Call fn, ignoring any exception that follows the deprecation warning."""
    try:
        fn()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Parameter deprecated aliases — warning emission
# ---------------------------------------------------------------------------

class TestParameterDeprecatedAliases:

    def test_declare_variable_warns(self):
        p = _make_input_stream_param("p_decl")
        with pytest.warns(DeprecationWarning) as w:
            _call_safely(p.declare_variable)
        _assert_deprecation(w, old="declare_variable", new="declare")

    def test_load_input_value_warns(self):
        p = _make_input_stream_param("p_rcv")
        _call_safely(p.declare)
        with pytest.warns(DeprecationWarning) as w:
            _call_safely(p.load_input_value)
        _assert_deprecation(w, old="load_input_value", new="rcv")

    def test_reset_var_warns(self):
        p = _make_input_stream_param("p_rst")
        _call_safely(p.declare)
        with pytest.warns(DeprecationWarning) as w:
            _call_safely(p.reset_var)
        _assert_deprecation(w, old="reset_var", new="reset_qua")

    def test_deprecated_alias_resolves_to_canonical(self):
        p = _make_input_stream_param("p_res")
        with pytest.warns(DeprecationWarning):
            method = p.declare_variable  # access only, don't call
        # The descriptor returns the canonical bound method
        assert method == p.declare

    # Canonical names must NOT be _DeprecatedAlias descriptors
    def test_canonical_declare_is_not_deprecated(self):
        assert not isinstance(Parameter.__dict__["declare"], _DeprecatedAlias)

    def test_canonical_rcv_is_not_deprecated(self):
        assert not isinstance(Parameter.__dict__["rcv"], _DeprecatedAlias)

    def test_canonical_reset_qua_is_not_deprecated(self):
        assert not isinstance(Parameter.__dict__["reset_qua"], _DeprecatedAlias)

    def test_canonical_declare_stream_is_not_deprecated(self):
        assert not isinstance(Parameter.__dict__["declare_stream"], _DeprecatedAlias)

    # Old names must BE _DeprecatedAlias descriptors
    def test_old_declare_variable_is_deprecated_alias(self):
        assert isinstance(Parameter.__dict__["declare_variable"], _DeprecatedAlias)

    def test_old_load_input_value_is_deprecated_alias(self):
        assert isinstance(Parameter.__dict__["load_input_value"], _DeprecatedAlias)

    def test_old_reset_var_is_deprecated_alias(self):
        assert isinstance(Parameter.__dict__["reset_var"], _DeprecatedAlias)


# ---------------------------------------------------------------------------
# ParameterTable deprecated aliases — warning emission
# ---------------------------------------------------------------------------

class TestParameterTableDeprecatedAliases:

    def test_declare_variables_warns(self):
        t = _make_input_stream_table()
        with pytest.warns(DeprecationWarning) as w:
            _call_safely(t.declare_variables)
        _assert_deprecation(w, old="declare_variables", new="declare")

    def test_load_input_values_warns(self):
        t = _make_input_stream_table(("lv1", "lv2"))
        _call_safely(t.declare)
        with pytest.warns(DeprecationWarning) as w:
            _call_safely(t.load_input_values)
        _assert_deprecation(w, old="load_input_values", new="rcv")

    def test_declare_streams_warns(self):
        t = _make_input_stream_table(("ds1", "ds2"))
        with pytest.warns(DeprecationWarning) as w:
            _call_safely(t.declare_streams)
        _assert_deprecation(w, old="declare_streams", new="declare_stream")

    def test_reset_vars_warns(self):
        t = _make_input_stream_table(("rv1", "rv2"))
        _call_safely(t.declare)
        with pytest.warns(DeprecationWarning) as w:
            _call_safely(t.reset_vars)
        _assert_deprecation(w, old="reset_vars", new="reset_qua")

    def test_deprecated_alias_resolves_to_canonical(self):
        t = _make_input_stream_table(("d1", "d2"))
        with pytest.warns(DeprecationWarning):
            method = t.declare_variables  # access only
        assert method == t.declare

    # Canonical names must NOT be _DeprecatedAlias descriptors
    def test_canonical_declare_is_not_deprecated(self):
        assert not isinstance(ParameterTable.__dict__["declare"], _DeprecatedAlias)

    def test_canonical_rcv_is_not_deprecated(self):
        assert not isinstance(ParameterTable.__dict__["rcv"], _DeprecatedAlias)

    def test_canonical_declare_stream_is_not_deprecated(self):
        assert not isinstance(ParameterTable.__dict__["declare_stream"], _DeprecatedAlias)

    def test_canonical_reset_qua_is_not_deprecated(self):
        assert not isinstance(ParameterTable.__dict__["reset_qua"], _DeprecatedAlias)

    # Old names must BE _DeprecatedAlias descriptors
    def test_old_declare_variables_is_deprecated_alias(self):
        assert isinstance(ParameterTable.__dict__["declare_variables"], _DeprecatedAlias)

    def test_old_load_input_values_is_deprecated_alias(self):
        assert isinstance(ParameterTable.__dict__["load_input_values"], _DeprecatedAlias)

    def test_old_declare_streams_is_deprecated_alias(self):
        assert isinstance(ParameterTable.__dict__["declare_streams"], _DeprecatedAlias)

    def test_old_reset_vars_is_deprecated_alias(self):
        assert isinstance(ParameterTable.__dict__["reset_vars"], _DeprecatedAlias)
