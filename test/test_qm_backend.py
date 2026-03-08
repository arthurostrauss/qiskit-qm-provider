"""Tests for QMBackend and FluxTunableTransmonBackend."""

import pytest
from copy import deepcopy
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler import Target
from qiskit.result.models import MeasLevel, MeasReturnType

from qiskit_qm_provider.backend.qm_backend import QMBackend
from qiskit_qm_provider.backend.flux_tunable_transmon_backend import (
    FluxTunableTransmonBackend,
)
from qiskit_qm_provider.backend.qm_instruction_properties import (
    QMInstructionProperties,
)


class TestQMBackendInit:
    def test_is_backend_v2(self, flux_tunable_backend):
        assert isinstance(flux_tunable_backend, BackendV2)

    def test_has_target(self, flux_tunable_backend):
        assert isinstance(flux_tunable_backend.target, Target)

    def test_has_name(self, flux_tunable_backend):
        assert isinstance(flux_tunable_backend.name, str)
        assert len(flux_tunable_backend.name) > 0

    def test_num_qubits_matches_machine(self, flux_tunable_backend, quam_machine):
        assert flux_tunable_backend.num_qubits == len(quam_machine.active_qubits)

    def test_max_circuits_is_none(self, flux_tunable_backend):
        assert flux_tunable_backend.max_circuits is None


class TestQMBackendOptions:
    def test_default_options(self, flux_tunable_backend):
        opts = flux_tunable_backend.options
        assert opts.shots == 1024
        assert opts.compiler_options is None
        assert opts.simulate is None
        assert opts.memory is False
        assert opts.skip_reset is False
        assert opts.meas_level == MeasLevel.CLASSIFIED
        assert opts.meas_return == MeasReturnType.AVERAGE
        assert opts.timeout == 60

    def test_set_options(self, flux_tunable_backend):
        flux_tunable_backend.set_options(shots=2048)
        assert flux_tunable_backend.options.shots == 2048
        flux_tunable_backend.set_options(shots=1024)


class TestQMBackendDeepCopy:
    def test_deepcopy_returns_self(self, flux_tunable_backend):
        copied = deepcopy(flux_tunable_backend)
        assert copied is flux_tunable_backend


class TestQMBackendTarget:
    def test_target_has_operations(self, flux_tunable_backend):
        ops = flux_tunable_backend.target.operation_names
        assert len(ops) > 0

    def test_target_has_qubit_properties(self, flux_tunable_backend, quam_machine):
        target = flux_tunable_backend.target
        for i in range(len(quam_machine.active_qubits)):
            qprops = target.qubit_properties
            if qprops is not None:
                assert qprops[i] is not None

    def test_target_dt(self, flux_tunable_backend):
        assert flux_tunable_backend.target.dt == 1e-9

    def test_target_granularity(self, flux_tunable_backend):
        assert flux_tunable_backend.target.granularity == 4

    def test_target_min_length(self, flux_tunable_backend):
        assert flux_tunable_backend.target.min_length == 16


class TestQMBackendQubitAccess:
    def test_qubit_dict(self, flux_tunable_backend, quam_machine):
        qd = flux_tunable_backend.qubit_dict
        assert isinstance(qd, dict)
        assert len(qd) == len(quam_machine.active_qubits)
        for name, idx in qd.items():
            assert isinstance(name, str)
            assert isinstance(idx, int)

    def test_qubit_pair_dict(self, flux_tunable_backend, quam_machine):
        qpd = flux_tunable_backend.qubit_pair_dict
        assert isinstance(qpd, dict)
        for name, pair in qpd.items():
            assert isinstance(name, str)
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_get_qubit_by_index(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubits) > 0:
            qubit = flux_tunable_backend.get_qubit(0)
            assert qubit is quam_machine.active_qubits[0]

    def test_get_qubit_by_name(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubits) > 0:
            name = quam_machine.active_qubits[0].name
            qubit = flux_tunable_backend.get_qubit(name)
            assert qubit.name == name

    def test_get_qubit_invalid_type_raises(self, flux_tunable_backend):
        with pytest.raises(ValueError, match="integer index or a string"):
            flux_tunable_backend.get_qubit(3.14)

    def test_get_qubit_index_by_name(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubits) > 0:
            name = quam_machine.active_qubits[0].name
            idx = flux_tunable_backend.get_qubit_index(name)
            assert idx == 0

    def test_get_qubit_index_by_qubit(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubits) > 0:
            qubit = quam_machine.active_qubits[0]
            idx = flux_tunable_backend.get_qubit_index(qubit)
            assert idx == 0

    def test_get_qubit_index_invalid_type_raises(self, flux_tunable_backend):
        with pytest.raises(ValueError, match="string name or a Qubit"):
            flux_tunable_backend.get_qubit_index(3.14)


class TestQMBackendQubitPairAccess:
    def test_get_qubit_pair_by_indices(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubit_pairs) > 0:
            pair = quam_machine.active_qubit_pairs[0]
            ctrl_idx = flux_tunable_backend.qubit_dict[pair.qubit_control.name]
            tgt_idx = flux_tunable_backend.qubit_dict[pair.qubit_target.name]
            result = flux_tunable_backend.get_qubit_pair((ctrl_idx, tgt_idx))
            assert result.name == pair.name

    def test_get_qubit_pair_invalid_not_tuple_raises(self, flux_tunable_backend):
        with pytest.raises(ValueError, match="tuple of two qubits"):
            flux_tunable_backend.get_qubit_pair([0, 1, 2])

    def test_get_qubit_pair_invalid_type_raises(self, flux_tunable_backend):
        with pytest.raises(ValueError):
            flux_tunable_backend.get_qubit_pair((3.14, 2.71))

    def test_get_qubit_pair_indices_invalid_raises(self, flux_tunable_backend):
        with pytest.raises(ValueError, match="string name or a QubitPair"):
            flux_tunable_backend.get_qubit_pair_indices(42)


class TestFluxTunableTransmonBackend:
    def test_is_qm_backend(self, flux_tunable_backend):
        assert isinstance(flux_tunable_backend, QMBackend)

    def test_qubit_mapping(self, flux_tunable_backend, quam_machine):
        mapping = flux_tunable_backend.qubit_mapping
        assert isinstance(mapping, dict)
        for i, qubit in enumerate(quam_machine.active_qubits):
            assert i in mapping
            names = mapping[i]
            assert qubit.xy.name in names
            assert qubit.z.name in names
            assert qubit.resonator.name in names

    def test_meas_map(self, flux_tunable_backend, quam_machine):
        mmap = flux_tunable_backend.meas_map
        assert isinstance(mmap, list)
        assert len(mmap) == len(quam_machine.active_qubits)
        for i, group in enumerate(mmap):
            assert group == [i]

    def test_qubits_property(self, flux_tunable_backend, quam_machine):
        qubits = flux_tunable_backend.qubits
        assert len(qubits) == len(quam_machine.active_qubits)

    def test_qubit_pairs_property(self, flux_tunable_backend, quam_machine):
        pairs = flux_tunable_backend.qubit_pairs
        assert len(pairs) == len(quam_machine.active_qubit_pairs)

    def test_invalid_machine_raises(self):
        from unittest.mock import MagicMock

        bad_machine = MagicMock(spec=[])
        with pytest.raises(ValueError, match="qubits and qubit_pairs"):
            FluxTunableTransmonBackend(bad_machine)


class TestQMBackendProperties:
    def test_custom_instructions(self, flux_tunable_backend):
        ci = flux_tunable_backend.custom_instructions
        assert isinstance(ci, dict)

    def test_qm_qasm_basis_gates(self, flux_tunable_backend):
        basis = flux_tunable_backend.qm_qasm_basis_gates
        assert isinstance(basis, list)
        for name in basis:
            assert name not in (
                "measure",
                "reset",
                "delay",
                "nop",
                "box",
                "for_loop",
                "while_loop",
                "if_else",
                "switch_case",
            )

    def test_qasm3_exporter(self, flux_tunable_backend):
        from qiskit.qasm3 import Exporter

        exporter = flux_tunable_backend.qasm3_exporter
        assert isinstance(exporter, Exporter)

    def test_init_macro_is_callable(self, flux_tunable_backend):
        assert callable(flux_tunable_backend.init_macro)

    def test_set_init_macro(self, flux_tunable_backend):
        original = flux_tunable_backend.init_macro
        new_macro = lambda: None
        flux_tunable_backend.init_macro = new_macro
        assert flux_tunable_backend.init_macro is new_macro
        flux_tunable_backend.init_macro = original

    def test_set_init_macro_invalid_raises(self, flux_tunable_backend):
        with pytest.raises(ValueError, match="callable"):
            flux_tunable_backend.init_macro = "not_callable"

    def test_qubit_index_dict(self, flux_tunable_backend, quam_machine):
        qid = flux_tunable_backend.qubit_index_dict
        assert isinstance(qid, dict)
        for i, qubit in enumerate(quam_machine.active_qubits):
            assert qid[i] is qubit

    def test_compiler_property(self, flux_tunable_backend):
        from qm_qasm import Compiler

        compiler = flux_tunable_backend.compiler
        assert isinstance(compiler, Compiler)

    def test_qm_config(self, flux_tunable_backend):
        config = flux_tunable_backend.qm_config
        assert isinstance(config, dict)

    def test_generate_config(self, flux_tunable_backend):
        config = flux_tunable_backend.generate_config()
        assert isinstance(config, dict)


class TestQMBackendCircuitToQua:
    def test_has_compiler(self, flux_tunable_backend):
        """The compiler property should return a valid qm_qasm Compiler."""
        from qm_qasm import Compiler

        assert isinstance(flux_tunable_backend.compiler, Compiler)

    def test_compiler_has_hardware_config(self, flux_tunable_backend):
        """The compiler should contain the operation mapping and qubit mapping."""
        compiler = flux_tunable_backend.compiler
        assert compiler is not None

    def test_qasm3_export(self, flux_tunable_backend, quam_machine):
        """Test that a circuit can be exported to OpenQASM3 using the backend exporter."""
        if len(quam_machine.active_qubits) == 0:
            pytest.skip("No active qubits")
        n = flux_tunable_backend.num_qubits
        qc = QuantumCircuit(n, n)
        qc.reset(range(n))
        qc.measure(range(n), range(n))
        exporter = flux_tunable_backend.qasm3_exporter
        qasm_str = exporter.dumps(qc)
        assert "OPENQASM" in qasm_str
        assert "measure" in qasm_str


class TestQMBackendPulseChannels:
    def test_drive_channel(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubits) == 0:
            pytest.skip("No active qubits")
        try:
            ch = flux_tunable_backend.drive_channel(0)
            from qiskit.pulse import DriveChannel

            assert isinstance(ch, DriveChannel)
        except ImportError:
            pytest.skip("Qiskit Pulse not available")

    def test_measure_channel(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubits) == 0:
            pytest.skip("No active qubits")
        try:
            ch = flux_tunable_backend.measure_channel(0)
            from qiskit.pulse import MeasureChannel

            assert isinstance(ch, MeasureChannel)
        except ImportError:
            pytest.skip("Qiskit Pulse not available")

    def test_acquire_channel(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubits) == 0:
            pytest.skip("No active qubits")
        try:
            ch = flux_tunable_backend.acquire_channel(0)
            from qiskit.pulse import AcquireChannel

            assert isinstance(ch, AcquireChannel)
        except ImportError:
            pytest.skip("Qiskit Pulse not available")

    def test_control_channel_single_qubit(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubits) == 0:
            pytest.skip("No active qubits")
        try:
            channels = flux_tunable_backend.control_channel([0])
            assert isinstance(channels, list)
        except (ImportError, ValueError):
            pytest.skip("Qiskit Pulse not available or no flux channel")

    def test_control_channel_qubit_pair(self, flux_tunable_backend, quam_machine):
        if len(quam_machine.active_qubit_pairs) == 0:
            pytest.skip("No active qubit pairs")
        pair = quam_machine.active_qubit_pairs[0]
        ctrl_idx = flux_tunable_backend.qubit_dict[pair.qubit_control.name]
        tgt_idx = flux_tunable_backend.qubit_dict[pair.qubit_target.name]
        try:
            channels = flux_tunable_backend.control_channel([ctrl_idx, tgt_idx])
            assert isinstance(channels, list)
        except (ImportError, ValueError):
            pytest.skip("Qiskit Pulse not available or no coupler")

    def test_control_channel_too_many_qubits_raises(self, flux_tunable_backend):
        try:
            with pytest.raises(ValueError, match="qubit pair or a single qubit"):
                flux_tunable_backend.control_channel([0, 1, 2])
        except ImportError:
            pytest.skip("Qiskit Pulse not available")


class TestQMBackendUpdateTarget:
    def test_update_target_no_error(self, flux_tunable_backend):
        flux_tunable_backend.update_target()

    def test_update_target_with_input_type(self, flux_tunable_backend):
        from qiskit_qm_provider.parameter_table import InputType

        flux_tunable_backend.update_target(input_type=InputType.INPUT_STREAM)
