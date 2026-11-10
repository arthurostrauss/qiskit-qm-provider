"""Tests for QMInstructionProperties."""

import inspect
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from qiskit.circuit.library import XGate
from qiskit.transpiler import InstructionProperties, Target

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
        assert prop.fidelity_dict is None

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

    def test_quam_macro_argument_is_supported(self):
        from quam.core.macro import QuamMacro

        mock_quam = MagicMock(spec=QuamMacro)
        mock_quam.duration = 80e-9
        mock_quam.fidelity = 0.98
        mock_quam.apply = MagicMock()
        prop = QMInstructionProperties(quam_macro=mock_quam)
        assert prop.quam_macro is mock_quam
        assert prop.qua_pulse_macro is mock_quam.apply
        assert prop.duration == 80e-9
        assert abs(prop.error - 0.02) < 1e-10

    def test_quam_macro_has_priority_over_qua_pulse_macro(self):
        from quam.core.macro import QuamMacro

        mock_quam = MagicMock(spec=QuamMacro)
        mock_quam.duration = 70e-9
        mock_quam.fidelity = 0.97
        mock_quam.apply = MagicMock()
        plain_callable = lambda: None
        prop = QMInstructionProperties(qua_pulse_macro=plain_callable, quam_macro=mock_quam)
        assert prop.quam_macro is mock_quam
        assert prop.qua_pulse_macro is mock_quam.apply
        assert prop.duration == 70e-9

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
        assert len(state) == 3

        new_prop = QMInstructionProperties.__new__(QMInstructionProperties)
        new_prop.__setstate__(state)
        assert new_prop._qua_pulse_macro is macro
        assert new_prop._explicit_fidelity_dict is None

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


def _target_prints_extra_properties() -> bool:
    source = inspect.getsource(Target.__str__)
    return "Extra properties" in source and "properties" in source


class TestFidelityDict:
    def test_mapping_fidelity_does_not_set_error(self):
        mock_macro = MagicMock()
        mock_macro.duration = 100e-9
        mock_macro.fidelity = {"rb": 0.99, "xeb": 0.95}
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        assert prop.error is None
        assert prop.fidelity_dict == {"rb": 0.99, "xeb": 0.95}

    def test_quam_dict_fidelity(self):
        from quam.core.quam_classes import QuamDict

        mock_macro = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = QuamDict({"rb": 0.98})
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        assert prop.error is None
        assert prop.fidelity_dict == {"rb": 0.98}

    def test_explicit_fidelity_dict_without_macro(self):
        prop = QMInstructionProperties(fidelity_dict={"rb": 0.9})
        assert prop.fidelity_dict == {"rb": 0.9}
        assert prop.error is None

    def test_macro_mapping_wins_over_explicit(self):
        mock_macro = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = {"rb": 0.99}
        prop = QMInstructionProperties(
            qua_pulse_macro=mock_macro,
            fidelity_dict={"xeb": 0.5},
        )
        assert prop.fidelity_dict == {"rb": 0.99}

    def test_explicit_used_when_macro_fidelity_is_scalar(self):
        mock_macro = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = 0.99
        prop = QMInstructionProperties(
            qua_pulse_macro=mock_macro,
            fidelity_dict={"rb": 0.9},
        )
        assert abs(prop.error - 0.01) < 1e-10
        assert prop.fidelity_dict == {"rb": 0.9}

    def test_live_mapping_update_is_visible(self):
        mock_macro = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = {"rb": 0.9}
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        mock_macro.fidelity = {"rb": 0.95, "xeb": 0.91}
        assert prop.fidelity_dict == {"rb": 0.95, "xeb": 0.91}

    def test_int_scalar_fidelity_infers_error(self):
        mock_macro = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = 1
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        assert prop.error == 0.0
        assert prop.fidelity_dict is None

    def test_getstate_preserves_explicit_fidelity_dict(self):
        prop = QMInstructionProperties(duration=1e-6, fidelity_dict={"rb": 0.9})
        state = prop.__getstate__()
        new_prop = QMInstructionProperties.__new__(QMInstructionProperties)
        new_prop.__setstate__(state)
        assert new_prop.fidelity_dict == {"rb": 0.9}

    def test_setstate_accepts_legacy_two_tuple(self):
        macro = MagicMock()
        macro.duration = None
        macro.fidelity = None
        prop = QMInstructionProperties(duration=1e-6, error=0.01, qua_pulse_macro=macro)
        base, stored_macro, _ = prop.__getstate__()
        new_prop = QMInstructionProperties.__new__(QMInstructionProperties)
        new_prop.__setstate__((base, stored_macro))
        assert new_prop._qua_pulse_macro is macro
        assert new_prop._explicit_fidelity_dict is None


class TestPropertiesDisplay:
    def test_quam_macro_repr_in_properties(self):
        from quam.core.macro import QuamMacro

        mock_macro = MagicMock(spec=QuamMacro)
        mock_macro.apply = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = None
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        extra = prop.properties
        assert extra is not None
        assert "quam_macro" in extra
        assert "qua_pulse_macro" not in extra
        assert extra["quam_macro"] == repr(mock_macro)

    def test_callable_qualname_not_function_repr(self):
        def my_qua_macro(theta):
            return None

        prop = QMInstructionProperties(qua_pulse_macro=my_qua_macro)
        extra = prop.properties
        assert extra is not None
        displayed = extra["qua_pulse_macro"]
        assert displayed.endswith("my_qua_macro")
        assert "<function" not in displayed
        assert "0x" not in displayed

    def test_empty_fidelity_dict_omitted_from_properties(self):
        mock_macro = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = {}
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        assert prop.fidelity_dict == {}
        extra = prop.properties
        if extra is not None:
            assert "fidelity_dict" not in extra
            assert "rb" not in extra

    def test_nonempty_fidelity_dict_unrolled_in_properties(self):
        mock_macro = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = {"rb": 0.99, "xeb": 0.95}
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        extra = prop.properties
        assert extra["rb"] == 0.99
        assert extra["xeb"] == 0.95
        assert "fidelity_dict" not in extra

    def test_reserved_fidelity_key_is_prefixed(self):
        mock_macro = MagicMock()
        mock_macro.duration = None
        mock_macro.fidelity = {"qua_pulse_macro": 0.9}
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        extra = prop.properties
        assert extra["fidelity.qua_pulse_macro"] == 0.9
        assert extra["qua_pulse_macro"] != 0.9

    def test_empty_properties_is_none(self):
        prop = QMInstructionProperties()
        assert prop.properties is None

    @pytest.mark.skipif(
        not _target_prints_extra_properties(),
        reason="this Qiskit Target.__str__ does not render Extra properties",
    )
    def test_target_str_includes_extra_properties(self):
        mock_macro = MagicMock()
        mock_macro.duration = 100e-9
        mock_macro.fidelity = {"rb": 0.99}
        prop = QMInstructionProperties(qua_pulse_macro=mock_macro)
        target = Target()
        target.add_instruction(XGate(), {(0,): prop})
        text = str(target)
        assert "Extra properties" in text
        assert "rb" in text
        assert "fidelity_dict" not in text


class TestDummyQuamCzFidelityDict:
    def test_cz_macro_injected_benchmarks(self):
        pytest.importorskip("quam_builder.builder.superconducting.modify_quam")
        from quam_builder.architecture.superconducting.qpu import FluxTunableQuam

        from test.dummy_quam import build_dummy_quam_for_qubits

        machine = build_dummy_quam_for_qubits(2, FluxTunableQuam)
        pair = machine.active_qubit_pairs[0]
        cz = pair.macros["cz"]
        cz.fidelity["rb"] = 0.99
        cz.fidelity["xeb"] = 0.95
        prop = QMInstructionProperties(qua_pulse_macro=cz)
        assert prop.error is None
        assert prop.fidelity_dict == {"rb": 0.99, "xeb": 0.95}
        extra = prop.properties
        assert extra is not None
        assert extra["rb"] == 0.99
        assert extra["xeb"] == 0.95
        assert "fidelity_dict" not in extra
        assert "quam_macro" in extra
        assert extra["quam_macro"].startswith("CZGate(")
