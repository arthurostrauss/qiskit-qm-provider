"""Tests for QMSamplerV2, QMEstimatorV2, and their options."""

import pytest
from qiskit.circuit import QuantumCircuit

from qiskit_qm_provider.primitives.qm_sampler import QMSamplerV2, QMSamplerOptions
from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorV2, QMEstimatorOptions
from qiskit_qm_provider.parameter_table.input_type import InputType


class TestQMSamplerOptions:
    def test_defaults(self):
        opts = QMSamplerOptions()
        assert opts.default_shots == 1024
        assert opts.input_type is None
        assert opts.run_options is None
        assert opts.meas_level == "classified"

    def test_custom_shots(self):
        opts = QMSamplerOptions(default_shots=4096)
        assert opts.default_shots == 4096

    def test_input_type_enum(self):
        opts = QMSamplerOptions(input_type=InputType.INPUT_STREAM)
        assert opts.input_type == InputType.INPUT_STREAM

    def test_input_type_string_coercion(self):
        opts = QMSamplerOptions(input_type="INPUT_STREAM")
        assert opts.input_type == InputType.INPUT_STREAM

    def test_invalid_input_type_raises(self):
        with pytest.raises(TypeError, match="InputType"):
            QMSamplerOptions(input_type=42)

    def test_run_options_dict(self):
        opts = QMSamplerOptions(run_options={"simulate": True})
        assert opts.run_options == {"simulate": True}

    def test_invalid_run_options_raises(self):
        with pytest.raises(TypeError, match="dictionary"):
            QMSamplerOptions(run_options="not_a_dict")

    def test_meas_level_options(self):
        for level in ("classified", "kerneled", "avg_kerneled"):
            opts = QMSamplerOptions(meas_level=level)
            assert opts.meas_level == level


class TestQMEstimatorOptions:
    def test_defaults(self):
        opts = QMEstimatorOptions()
        assert opts.default_precision == pytest.approx(0.015625)
        assert opts.abelian_grouping is True
        assert opts.input_type is None
        assert opts.run_options is None

    def test_custom_precision(self):
        opts = QMEstimatorOptions(default_precision=0.01)
        assert opts.default_precision == 0.01

    def test_abelian_grouping(self):
        opts = QMEstimatorOptions(abelian_grouping=False)
        assert opts.abelian_grouping is False

    def test_input_type_enum(self):
        opts = QMEstimatorOptions(input_type=InputType.DGX_Q)
        assert opts.input_type == InputType.DGX_Q

    def test_input_type_string_coercion(self):
        opts = QMEstimatorOptions(input_type="IO1")
        assert opts.input_type == InputType.IO1

    def test_invalid_input_type_raises(self):
        with pytest.raises(TypeError, match="InputType"):
            QMEstimatorOptions(input_type=42)

    def test_run_options_dict(self):
        opts = QMEstimatorOptions(run_options={"shots": 2048})
        assert opts.run_options == {"shots": 2048}

    def test_invalid_run_options_raises(self):
        with pytest.raises(TypeError, match="dictionary"):
            QMEstimatorOptions(run_options=[1, 2, 3])

    def test_as_dict(self):
        opts = QMEstimatorOptions()
        d = opts.as_dict()
        assert isinstance(d, dict)
        assert "default_precision" in d
        assert "abelian_grouping" in d
        assert "input_type" in d
        assert "run_options" in d


class TestQMSamplerV2Init:
    def test_init_with_defaults(self, flux_tunable_backend):
        sampler = QMSamplerV2(flux_tunable_backend)
        assert sampler.backend is flux_tunable_backend
        assert isinstance(sampler.options, QMSamplerOptions)
        assert sampler.options.default_shots == 1024

    def test_init_with_options_object(self, flux_tunable_backend):
        opts = QMSamplerOptions(default_shots=512)
        sampler = QMSamplerV2(flux_tunable_backend, options=opts)
        assert sampler.options.default_shots == 512

    def test_init_with_dict_options(self, flux_tunable_backend):
        sampler = QMSamplerV2(flux_tunable_backend, options={"default_shots": 2048})
        assert sampler.options.default_shots == 2048

    def test_init_with_none_options(self, flux_tunable_backend):
        sampler = QMSamplerV2(flux_tunable_backend, options=None)
        assert sampler.options.default_shots == 1024


class TestQMEstimatorV2Init:
    def test_init_with_defaults(self, flux_tunable_backend):
        estimator = QMEstimatorV2(flux_tunable_backend)
        assert estimator.backend is flux_tunable_backend
        assert isinstance(estimator.options, QMEstimatorOptions)

    def test_init_with_options_object(self, flux_tunable_backend):
        opts = QMEstimatorOptions(default_precision=0.01)
        estimator = QMEstimatorV2(flux_tunable_backend, options=opts)
        assert estimator.options.default_precision == 0.01

    def test_init_with_dict_options(self, flux_tunable_backend):
        estimator = QMEstimatorV2(
            flux_tunable_backend, options={"default_precision": 0.005}
        )
        assert estimator.options.default_precision == 0.005

    def test_init_creates_switch_obs_circuit(self, flux_tunable_backend):
        estimator = QMEstimatorV2(flux_tunable_backend)
        assert estimator._switch_obs_circuit is not None
        assert isinstance(estimator._switch_obs_circuit, QuantumCircuit)

    def test_init_creates_passmanager(self, flux_tunable_backend):
        from qiskit.transpiler import PassManager

        estimator = QMEstimatorV2(flux_tunable_backend)
        assert isinstance(estimator._passmanager, PassManager)


class TestQMSamplerV2ValidatePubs:
    def test_validate_pubs_no_creg_warns(self, flux_tunable_backend):
        sampler = QMSamplerV2(flux_tunable_backend)
        n = flux_tunable_backend.num_qubits
        qc = QuantumCircuit(n)
        qc.reset(0)
        qc.x(0)

        from qiskit.primitives.containers.sampler_pub import SamplerPub

        pub = SamplerPub.coerce(qc, 1024)
        with pytest.warns(UserWarning, match="no output classical registers"):
            sampler._validate_pubs([pub])

    def test_validate_pubs_adds_reset(self, flux_tunable_backend):
        sampler = QMSamplerV2(flux_tunable_backend)
        n = flux_tunable_backend.num_qubits
        qc = QuantumCircuit(n, n)
        qc.x(0)
        qc.measure(range(n), range(n))

        from qiskit.primitives.containers.sampler_pub import SamplerPub

        pub = SamplerPub.coerce(qc, 1024)
        result = sampler._validate_pubs([pub])
        assert len(result) == 1
        ops = [inst.operation.name for inst in result[0].circuit.data]
        assert "reset" in ops


class TestQMEstimatorV2ValidatePubs:
    def test_validate_invalid_precision_raises(self, flux_tunable_backend):
        """Negative precision is rejected during pub coercion."""
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives.containers.estimator_pub import EstimatorPub

        n = flux_tunable_backend.num_qubits
        qc = QuantumCircuit(n)
        obs = SparsePauliOp.from_list([("I" * n, 1.0)])

        with pytest.raises(ValueError, match="precision"):
            EstimatorPub.coerce((qc, obs), precision=-1.0)

    def test_validate_wrong_num_qubits_raises(self, flux_tunable_backend):
        estimator = QMEstimatorV2(flux_tunable_backend)
        n = flux_tunable_backend.num_qubits
        qc = QuantumCircuit(n + 1)  # wrong number of qubits
        from qiskit.quantum_info import SparsePauliOp

        obs = SparsePauliOp.from_list([("I" * (n + 1), 1.0)])

        from qiskit.primitives.containers.estimator_pub import EstimatorPub

        pub = EstimatorPub.coerce((qc, obs), precision=0.01)
        with pytest.raises(ValueError):
            estimator.validate_estimator_pubs([pub])
