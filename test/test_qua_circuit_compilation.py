"""Tests for QuaCircuitCompilation and measurement output wiring."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qiskit.circuit import QuantumCircuit, ClassicalRegister, Clbit
from qm import qua
from qm.qua._expressions import QuaArrayVariable

from qiskit_qm_provider import Parameter, ParameterPool, ParameterTable
from qiskit_qm_provider.backend.backend_utils import get_measurement_outcomes
from qiskit_qm_provider.backend.measurement_field import MeasurementRegisterField
from qiskit_qm_provider.backend.qua_circuit_compilation import (
    MeasurementOutcomeTable,
    QuaCircuitCompilation,
)


@pytest.fixture(autouse=True)
def reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


class MockResultProgram:
    def __init__(self, mapping: dict):
        self._mapping = mapping
        self.dsl_program = object()

    def __getitem__(self, key: str):
        return self._mapping[key]


def _mock_compilation_result(mapping: dict, name: str = "test_comp"):
    return SimpleNamespace(
        name=name,
        result_program=MockResultProgram(mapping),
        uuid="00000000-0000-0000-0000-000000000001",
        source_code="OPENQASM 3.0;",
    )


class TestQuaCircuitCompilationDelegation:
    def test_delegates_compilation_result_attributes(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False, False])

        mapping = {"c": array_var}
        result = _mock_compilation_result(mapping)
        creg = ClassicalRegister(2, "c")
        qc = QuantumCircuit(2)
        qc.add_register(creg)
        qc.measure([0, 1], creg)
        wrapper = QuaCircuitCompilation(result, qc)

        assert wrapper.result_program is result.result_program
        assert wrapper.qua_program is result.result_program.dsl_program
        assert wrapper.name == "test_comp"
        assert wrapper.compilation_result is result
        assert wrapper.outputs.name.endswith("_output")


class TestMeasurementOutcomeTableAccessors:
    def test_getitem_returns_qua_var_inside_program(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False, False])
            loose0 = qua.declare(bool, value=False)
            loose1 = qua.declare(bool, value=False)

        qc = QuantumCircuit(2, name="meas_qc")
        creg = ClassicalRegister(2, "c")
        loose_clbit0 = Clbit()
        loose_clbit1 = Clbit()
        qc.add_register(creg)
        qc.add_bits([loose_clbit0, loose_clbit1])
        qc.measure(0, creg[0])
        qc.measure(1, creg[1])
        qc.measure(0, loose_clbit0)
        qc.measure(1, loose_clbit1)

        result = _mock_compilation_result(
            {
                "c": array_var,
                "_bit0": loose0,
                "_bit1": loose1,
            }
        )
        table = MeasurementOutcomeTable.from_compilation(qc, result)

        assert table.name == "meas_qc_output"
        assert "c" in table.table
        assert "_bit0" in table.table
        assert "_bit1" in table.table

        with qua.program():
            assert table["c"] is array_var
            assert isinstance(table["c"], QuaArrayVariable)
            field = table.get_parameter("c")
            assert isinstance(field, MeasurementRegisterField)
            assert field.size == 2
            assert table.state_ints["c"] is not None
            assert table.streams["c"] is not None

    def test_getitem_outside_program_raises(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False, False])

        creg = ClassicalRegister(2, "c")
        qc = QuantumCircuit(2)
        qc.add_register(creg)
        qc.measure([0, 1], creg)
        result = _mock_compilation_result({"c": array_var})
        table = MeasurementOutcomeTable.from_compilation(qc, result)

        with pytest.raises(RuntimeError, match="with program\\(\\):"):
            _ = table["c"]

    def test_not_registered_in_runtime_parameter_pool(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False])

        creg = ClassicalRegister(1, "meas")
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _mock_compilation_result({"meas": array_var})

        table = MeasurementOutcomeTable.from_compilation(qc, result)
        assert table._id == 0
        assert table not in ParameterPool.get_all_objs()

    def test_tracked_in_measurement_registries(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False])
            loose = qua.declare(bool, value=False)

        qc = QuantumCircuit(2)
        creg = ClassicalRegister(1, "c")
        loose_bit = Clbit()
        qc.add_register(creg)
        qc.add_bits([loose_bit])
        qc.measure(0, creg[0])
        qc.measure(1, loose_bit)
        result = _mock_compilation_result({"c": array_var, "_bit0": loose})

        table = MeasurementOutcomeTable.from_compilation(qc, result)
        assert table in ParameterPool.iter_measurement_outcome_tables()
        fields = list(ParameterPool.iter_measurement_register_fields())
        assert table.get_parameter("c") in fields
        assert table.get_parameter("_bit0") in fields

    def test_size_one_creg_var_is_array(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False])

        creg = ClassicalRegister(1, "b")
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _mock_compilation_result({"b": array_var})
        table = MeasurementOutcomeTable.from_compilation(qc, result)

        with qua.program():
            assert table.get_parameter("b").size == 1
            assert isinstance(table["b"], QuaArrayVariable)
            assert table["b"] is array_var

    def test_declare_is_noop(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False])

        creg = ClassicalRegister(1, "c")
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _mock_compilation_result({"c": array_var})
        table = MeasurementOutcomeTable.from_compilation(qc, result)

        with qua.program():
            returned = table.declare()
            assert returned is array_var


class TestMeasurementRegisterField:
    def test_state_int_size_one_creg_uses_array_indexing(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False])
            scalar_var = qua.declare(bool, value=False)

        creg = ClassicalRegister(1, "c")
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.measure(0, creg[0])

        creg_field = MeasurementRegisterField("c", 1)
        creg_field._wire_from_result(_mock_compilation_result({"c": array_var}), "c")

        loose_field = MeasurementRegisterField("_bit0", 1)
        loose_field._wire_from_result(_mock_compilation_result({"_bit0": scalar_var}), "_bit0")

        with qua.program():
            creg_state_int = creg_field.state_int
            loose_state_int = loose_field.state_int
            assert creg_state_int is not None
            assert loose_state_int is not None

    def test_from_compilation_records_var_shape(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False])
            loose_var = qua.declare(bool, value=False)

        creg = ClassicalRegister(1, "c")
        loose_clbit = Clbit()
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.add_bits([loose_clbit])
        qc.measure(0, creg[0])
        qc.measure(0, loose_clbit)

        result = _mock_compilation_result({"c": array_var, "_bit0": loose_var})
        table = MeasurementOutcomeTable.from_compilation(qc, result)

        assert table.get_parameter("c").var_is_array is True
        assert table.get_parameter("_bit0").var_is_array is False

        with qua.program():
            assert table.state_ints["c"] is not None
            assert table.state_ints["_bit0"] is not None

    def test_wire_multi_bit_scalar_raises(self):
        with qua.program():
            scalar = qua.declare(bool, value=False)

        field = MeasurementRegisterField("c", 2)
        with pytest.raises(ValueError, match="packed output"):
            field._wire_from_result(_mock_compilation_result({"c": scalar}), "c", register_size=2)

    def test_rewire_invalidates_state_int_on_size_change(self):
        with qua.program():
            var1 = qua.declare(bool, value=False)
            var2 = qua.declare(bool, value=[False, False])

        field = MeasurementRegisterField("c", 1)
        result1 = _mock_compilation_result({"c": var1}, name="r1")
        result2 = _mock_compilation_result(
            {"c": var2},
            name="r2",
        )
        result2.uuid = "00000000-0000-0000-0000-000000000002"

        field._wire_from_result(result1, "c", register_size=1)
        with qua.program():
            field._state_int_var = MagicMock(name="state_int_var")
            field._wire_from_result(result2, "c", register_size=2)

        assert field._var is var2
        assert field._state_int_var is None
        assert field._stream is None
        assert field.size == 2

    def test_new_compilation_yields_new_field_objects(self):
        with qua.program():
            var = qua.declare(bool, value=False)

        creg = ClassicalRegister(1, "c")
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _mock_compilation_result({"c": var})

        table1 = MeasurementOutcomeTable.from_compilation(qc, result)
        table2 = MeasurementOutcomeTable.from_compilation(qc, result)
        assert table1 is not table2
        assert table1.get_parameter("c") is not table2.get_parameter("c")

    def test_rewire_on_same_compilation_refreshes_var(self):
        with qua.program():
            var1 = qua.declare(bool, value=False)
            var2 = qua.declare(bool, value=False)

        creg = ClassicalRegister(1, "c")
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result1 = _mock_compilation_result({"c": var1})
        result2 = _mock_compilation_result({"c": var2})
        result2.uuid = "00000000-0000-0000-0000-000000000002"

        comp = QuaCircuitCompilation(result1, qc)
        field = comp.outputs.get_parameter("c")
        comp.rewire_outputs(qc, result2)
        assert field._var is var2

    def test_cannot_attach_to_runtime_parameter_table(self):
        field = MeasurementRegisterField("c", 1)
        with pytest.raises(AssertionError, match="Parameter object"):
            ParameterTable([field], name="Bad")

    def test_runtime_table_rejects_measurement_field_via_add_parameters(self):
        field = MeasurementRegisterField("c", 1)
        runtime_table = ParameterTable([], name="Empty")
        with pytest.raises(ValueError, match="cannot be attached"):
            runtime_table.add_parameters(field)

    def test_deepcopy_raises(self):
        field = MeasurementRegisterField("c", 1)
        with pytest.raises(TypeError, match="cannot be deepcopied"):
            deepcopy(field)

    def test_var_outside_program_raises(self):
        field = MeasurementRegisterField("c", 1)
        with pytest.raises(RuntimeError, match="with program\\(\\):"):
            _ = field.var


class TestRuntimeMeasurementNameCoexistence:
    def test_runtime_and_measurement_same_name_coexist(self):
        ParameterTable({"c": 0.0}, name="Inputs")
        with qua.program():
            var = qua.declare(bool, value=[False])

        creg = ClassicalRegister(1, "c")
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _mock_compilation_result({"c": var})

        table = MeasurementOutcomeTable.from_compilation(qc, result)
        runtime = ParameterPool.lookup_runtime_parameter("c")
        assert runtime is not None
        assert not isinstance(runtime, MeasurementRegisterField)
        meas_field = table.get_parameter("c")
        assert meas_field is not runtime
        assert meas_field.name == runtime.name == "c"

    def test_pool_reset_clears_measurement_registries(self):
        with qua.program():
            var = qua.declare(bool, value=[False])

        creg = ClassicalRegister(1, "c")
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _mock_compilation_result({"c": var})

        table = MeasurementOutcomeTable.from_compilation(qc, result)
        field = table.get_parameter("c")
        assert list(ParameterPool.iter_measurement_outcome_tables())
        assert list(ParameterPool.iter_measurement_register_fields())

        ParameterPool.reset()
        assert list(ParameterPool.iter_measurement_outcome_tables()) == []
        assert list(ParameterPool.iter_measurement_register_fields()) == []

        new_table = MeasurementOutcomeTable.from_compilation(qc, result)
        assert new_table.get_parameter("c") is not field


class TestGetMeasurementOutcomesShim:
    def test_shim_from_qua_circuit_compilation(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False, False])

        creg = ClassicalRegister(2, "meas")
        qc = QuantumCircuit(2)
        qc.add_register(creg)
        qc.measure([0, 1], creg)
        result = _mock_compilation_result({"meas": array_var})

        wrapper = QuaCircuitCompilation(result, qc)

        with qua.program():
            meas = get_measurement_outcomes(qc, wrapper)

        assert "meas" in meas
        assert meas["meas"]["size"] == 2
        assert meas["meas"]["value"] is array_var
        assert "state_int" in meas["meas"]
        assert "stream" in meas["meas"]

    def test_outputs_attr_access_does_not_expose_state_int_on_var(self):
        with qua.program():
            array_var = qua.declare(bool, value=[False])

        creg = ClassicalRegister(1, "meas")
        qc = QuantumCircuit(1)
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _mock_compilation_result({"meas": array_var})
        wrapper = QuaCircuitCompilation(result, qc)

        with qua.program():
            var = wrapper.outputs["meas"]
            assert not hasattr(var, "state_int")
            assert wrapper.outputs.state_ints["meas"] is not None
