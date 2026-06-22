"""Tests for MeasurementRegisterField and scope guards."""

from __future__ import annotations

import pytest
from qm import qua

from qiskit_qm_provider import Parameter, ParameterPool, ParameterTable
from qiskit_qm_provider.backend.measurement_field import MeasurementRegisterField
from qiskit_qm_provider.parameter_table._scope import is_inside_scope


@pytest.fixture(autouse=True)
def reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


class TestScopeGuards:
    def test_parameter_var_outside_program_raises(self):
        table = ParameterTable({"x": 0.0}, name="T")
        with qua.program():
            table.declare()
        with pytest.raises(RuntimeError, match="with program\\(\\):"):
            _ = table.get_parameter("x").var

    def test_parameter_table_getitem_outside_program_raises(self):
        table = ParameterTable({"x": 0.0}, name="T")
        with qua.program():
            table.declare()
        with pytest.raises(RuntimeError, match="with program\\(\\):"):
            _ = table["x"]

    def test_is_inside_scope_tracks_program_block(self):
        assert not is_inside_scope()
        with qua.program():
            assert is_inside_scope()
        assert not is_inside_scope()

    def test_measurement_field_is_not_parameter(self):
        field = MeasurementRegisterField("c", 1)
        assert field.is_measurement_output
        assert not isinstance(field, Parameter)
