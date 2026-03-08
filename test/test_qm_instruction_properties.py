"""Tests for QMInstructionProperties."""

import pytest
from copy import deepcopy
from unittest.mock import MagicMock

from qiskit.transpiler import InstructionProperties
from qiskit_qm_provider.backend.qm_instruction_properties import (
    QMInstructionProperties,
)


class TestQMInstructionPropertiesInit:
    def test_basic_creation(self):
        prop = QMInstructionProperties()
        assert prop.duration is None
        assert prop.error is None
        assert prop.qua_pulse_macro is None

    def test_with_duration_and_error(self):
        prop = QMInstructionProperties(duration=1e-6, error=0.01)
        assert prop.duration == 1e-6
        assert prop.error == 0.01

    def test_with_callable_macro(self):
        macro = lambda: None
        prop = QMInstructionProperties(qua_pulse_macro=macro)
        assert prop.qua_pulse_macro is macro

    def test_with_quam_macro_duration_inference(self):
        mock_macro = MagicMock()
        mock_macro.duration = 100e-9
        mock_macro.fidelity = 0.99
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        assert prop.duration == 100e-9
        assert abs(prop.error - 0.01) < 1e-10

    def test_with_quam_macro_pulse_duration_inference(self):
        mock_pulse = MagicMock()
        mock_pulse.length = 40
        mock_macro = MagicMock()
        mock_macro.duration = None
        mock_macro.pulse = mock_pulse
        mock_macro.fidelity = "not_a_float"  # non-float fidelity
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        assert prop.duration == 40e-9

    def test_explicit_duration_overrides_macro(self):
        mock_macro = MagicMock()
        mock_macro.duration = 100e-9
        mock_macro.fidelity = 0.99
        prop = QMInstructionProperties(duration=200e-9, qua_pulse_macro=mock_macro)
        assert prop.duration == 200e-9

    def test_isinstance_instruction_properties(self):
        prop = QMInstructionProperties()
        assert isinstance(prop, InstructionProperties)


class TestQMInstructionPropertiesQuaMacro:
    def test_quam_macro_property_with_quam_macro(self):
        from quam.core.macro import QuamMacro

        mock_macro = MagicMock(spec=QuamMacro)
        mock_macro.apply = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = None
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        assert prop.qua_pulse_macro is mock_macro.apply
        assert prop.quam_macro is mock_macro

    def test_quam_macro_property_with_callable(self):
        fn = lambda: None
        prop = QMInstructionProperties(qua_pulse_macro=fn)
        assert prop.qua_pulse_macro is fn
        assert prop.quam_macro is None

    def test_set_qua_pulse_macro(self):
        prop = QMInstructionProperties()
        new_macro = lambda: "test"
        prop.qua_pulse_macro = new_macro
        assert prop.qua_pulse_macro is new_macro

    def test_set_quam_macro(self):
        from quam.core.macro import QuamMacro

        prop = QMInstructionProperties()
        mock_macro = MagicMock(spec=QuamMacro)
        mock_macro.apply = MagicMock()
        prop.quam_macro = mock_macro
        assert prop.qua_pulse_macro is mock_macro.apply


class TestQMInstructionPropertiesSerialisation:
    def test_repr(self):
        prop = QMInstructionProperties(duration=1e-6, error=0.01)
        r = repr(prop)
        assert "QMInstructionProperties" in r
        assert "duration" in r
        assert "error" in r

    def test_getstate_setstate_roundtrip(self):
        macro = MagicMock()
        macro.duration = None
        macro.fidelity = None
        prop = QMInstructionProperties(duration=1e-6, error=0.01, qua_pulse_macro=macro)
        state = prop.__getstate__()
        assert isinstance(state, tuple)
        assert len(state) == 2

        new_prop = QMInstructionProperties.__new__(QMInstructionProperties)
        new_prop.__setstate__(state)
        assert new_prop._qua_pulse_macro is macro

    def test_deepcopy_preserves_macro_ref(self):
        macro = MagicMock()
        macro.duration = None
        macro.fidelity = None
        prop = QMInstructionProperties(duration=1e-6, error=0.01, qua_pulse_macro=macro)
        prop_copy = deepcopy(prop)
        assert prop_copy._qua_pulse_macro is prop._qua_pulse_macro
        assert prop_copy.duration == prop.duration
        assert prop_copy.error == prop.error

    def test_deepcopy_memo_prevents_duplication(self):
        macro = MagicMock()
        macro.duration = None
        macro.fidelity = None
        prop = QMInstructionProperties(duration=1e-6, error=0.01, qua_pulse_macro=macro)
        memo = {}
        copy1 = deepcopy(prop, memo)
        copy2 = deepcopy(prop, memo)
        assert copy1 is copy2
