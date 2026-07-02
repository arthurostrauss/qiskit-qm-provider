"""Tests for Primitives chunking: plan_sampler_programs, plan_estimator_programs,
QMSamplerJob, and QMEstimatorJob pre/post-processing (up to program generation).

All tests stop before submitting to hardware (no QM connection required).
"""

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qm import Program

from qiskit_qm_provider.job.qua_programs import (
    compute_chunk_layout,
    plan_sampler_programs,
    plan_estimator_programs,
)
from qiskit_qm_provider.parameter_table import ParameterPool, ParameterTable
from qiskit_qm_provider.parameter_table.input_type import InputType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure_circuit(backend):
    """Measurement-only circuit transpiled to the backend's native gate set."""
    n = backend.num_qubits
    qc = QuantumCircuit(n, 1)
    qc.measure(0, 0)
    # Transpile so quantum_circuit_to_qua sees only native backend operations.
    return transpile(qc, backend=backend, optimization_level=0)


def _sampler_pubs(backend, count):
    """Return ``count`` coerced SamplerPubs for a transpiled measure circuit."""
    qc = _measure_circuit(backend)
    return [SamplerPub.coerce(qc, 64) for _ in range(count)]


def _estimator_pubs(backend, count):
    """Return ``count`` fully-validated EstimatorPubs ready for _ExecutionPlan.

    Each pub gets a circuit with a unique name so that ParameterTable names
    derived from it (e.g. "observables_var_{name}_pub{i}") are globally unique.
    This prevents Parameter.__new__'s name-based singleton from returning the
    same Parameter object across pubs, which would cause "already declared" errors
    when multiple plans are compiled inside a single QUA program() context.
    """
    from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorV2

    n = backend.num_qubits
    obs = SparsePauliOp.from_list([("Z" + "I" * (n - 1), 1.0)])
    # Unique circuit per pub → unique obs var names per plan ("obs_0" in "circuit_k")
    raw = []
    for k in range(count):
        qc = QuantumCircuit(n, name=f"est_circ_{k}")
        qc.id(0)
        qc = transpile(qc, backend=backend, optimization_level=0)
        raw.append(EstimatorPub.coerce((qc, obs), precision=0.05))
    estimator = QMEstimatorV2(backend, options={"input_type": InputType.INPUT_STREAM})
    return estimator.validate_estimator_pubs(raw)


def _param_tables(pubs, input_type=None):
    """Build per-pub ParameterTables (no parameters → empty tables)."""
    from qiskit.circuit import Parameter
    ParameterPool.reset()
    return [
        ParameterTable.from_qiskit(
            pub.circuit,
            input_type=input_type,
            filter_function=lambda x: isinstance(x, Parameter),
            name=f"pt_{i}",
        )
        for i, pub in enumerate(pubs)
    ]


def _execution_plans_and_obs_var(backend, count, input_type=InputType.INPUT_STREAM):
    """Return (pubs, plans, obs_length_var) for estimator tests.

    Plans are built after a single pool reset so Parameter.__new__'s singleton
    reuses the same 'obs_0' object across plans with the same circuit.  The
    warn-and-skip fix in declare_variable() (for non-DGX_Q streaming types)
    ensures the shared Parameter is declared once in the QUA program and
    skipped on subsequent plans.
    """
    from qiskit_qm_provider.job.qm_estimator_job import _ExecutionPlan
    from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions
    from qiskit_qm_provider.parameter_table import Parameter as QuaParameter
    pubs = _estimator_pubs(backend, count)
    ParameterPool.reset()
    opts = QMEstimatorOptions(input_type=input_type)
    plans = [_ExecutionPlan.from_pub(pub, opts) for pub in pubs]
    obs_length_var = QuaParameter(
        name="obs_length_var", value=0, qua_type=int, input_type=input_type
    )
    return pubs, plans, obs_length_var


def _locator(chunk_layout):
    return {g: (c, l) for c, chunk in enumerate(chunk_layout) for l, g in enumerate(chunk)}


# ---------------------------------------------------------------------------
# plan_sampler_programs
# ---------------------------------------------------------------------------

class TestPlanSamplerPrograms:

    def test_single_program_under_limit(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=10)
        pubs = _sampler_pubs(flux_tunable_backend, 3)
        tables = _param_tables(pubs)
        programs, layout = plan_sampler_programs(flux_tunable_backend, pubs, tables)
        assert layout == [[0, 1, 2]]
        assert len(programs) == 1
        assert isinstance(programs[0], Program)

    def test_splits_into_multiple_programs(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=3)
        pubs = _sampler_pubs(flux_tunable_backend, 7)
        tables = _param_tables(pubs)
        programs, layout = plan_sampler_programs(flux_tunable_backend, pubs, tables)
        assert layout == [[0, 1, 2], [3, 4, 5], [6]]
        assert len(programs) == 3
        assert all(isinstance(p, Program) for p in programs)

    def test_large_max_circuits_never_splits(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=100)
        pubs = _sampler_pubs(flux_tunable_backend, 5)
        tables = _param_tables(pubs)
        programs, layout = plan_sampler_programs(flux_tunable_backend, pubs, tables)
        assert len(programs) == 1
        assert layout == [list(range(5))]

    def test_layout_is_full_partition(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=3)
        n = 8
        pubs = _sampler_pubs(flux_tunable_backend, n)
        tables = _param_tables(pubs)
        _, layout = plan_sampler_programs(flux_tunable_backend, pubs, tables)
        flat = [g for chunk in layout for g in chunk]
        assert flat == list(range(n))

    def test_chunk_sizes_respect_max_circuits(self, flux_tunable_backend):
        max_c = 3
        flux_tunable_backend.set_options(max_circuits=max_c)
        pubs = _sampler_pubs(flux_tunable_backend, 10)
        tables = _param_tables(pubs)
        _, layout = plan_sampler_programs(flux_tunable_backend, pubs, tables)
        assert all(len(chunk) <= max_c for chunk in layout)


# ---------------------------------------------------------------------------
# plan_estimator_programs
# ---------------------------------------------------------------------------

class TestPlanEstimatorPrograms:

    def test_single_program_under_limit(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=10)
        _, plans, obs_var = _execution_plans_and_obs_var(flux_tunable_backend, 2)
        programs, layout = plan_estimator_programs(
            flux_tunable_backend, plans, obs_length_var=obs_var
        )
        assert layout == [[0, 1]]
        assert len(programs) == 1
        assert isinstance(programs[0], Program)

    def test_splits_into_multiple_programs(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=2)
        _, plans, obs_var = _execution_plans_and_obs_var(flux_tunable_backend, 5)
        programs, layout = plan_estimator_programs(
            flux_tunable_backend, plans, obs_length_var=obs_var
        )
        assert layout == [[0, 1], [2, 3], [4]]
        assert len(programs) == 3
        assert all(isinstance(p, Program) for p in programs)

    def test_large_max_circuits_never_splits(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=100)
        _, plans, obs_var = _execution_plans_and_obs_var(flux_tunable_backend, 4)
        programs, layout = plan_estimator_programs(
            flux_tunable_backend, plans, obs_length_var=obs_var
        )
        assert len(programs) == 1
        assert layout == [list(range(4))]

    def test_layout_is_full_partition(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=3)
        n = 7
        _, plans, obs_var = _execution_plans_and_obs_var(flux_tunable_backend, n)
        _, layout = plan_estimator_programs(
            flux_tunable_backend, plans, obs_length_var=obs_var
        )
        flat = [g for chunk in layout for g in chunk]
        assert flat == list(range(n))


# ---------------------------------------------------------------------------
# Locator correctness for Primitives (mirrors TestResultLocator for run path)
# ---------------------------------------------------------------------------

class TestPrimitivesLocator:

    @pytest.mark.parametrize("n,max_c", [(1, 5), (5, 5), (6, 5), (11, 5), (13, 4)])
    def test_locator_covers_all_global_indices(self, n, max_c):
        layout = compute_chunk_layout(n, max_circuits=max_c)
        loc = _locator(layout)
        assert sorted(loc.keys()) == list(range(n))

    @pytest.mark.parametrize("n,max_c", [(1, 5), (5, 5), (6, 5), (11, 5), (13, 4)])
    def test_locator_pairs_are_unique(self, n, max_c):
        layout = compute_chunk_layout(n, max_circuits=max_c)
        loc = _locator(layout)
        assert len(set(loc.values())) == n

    def test_locator_local_index_resets_per_chunk(self):
        # With max_circuits=3 and 7 items, global 3 must map to local 0 in chunk 1.
        layout = compute_chunk_layout(7, max_circuits=3)
        loc = _locator(layout)
        assert loc[3] == (1, 0)
        assert loc[6] == (2, 0)

    def test_single_program_locator_is_identity(self):
        layout = compute_chunk_layout(5, max_circuits=10)
        loc = _locator(layout)
        for g in range(5):
            assert loc[g] == (0, g)


# ---------------------------------------------------------------------------
# QMSamplerJob construction (no submit — checks _chunk_layout, _locator, _program)
# ---------------------------------------------------------------------------

class TestQMSamplerJobConstruction:

    def test_single_program_in_list(self, flux_tunable_backend):
        from qiskit_qm_provider.job.qm_sampler_job import QMSamplerJob

        flux_tunable_backend.set_options(max_circuits=10)
        pubs = _sampler_pubs(flux_tunable_backend, 3)
        job = QMSamplerJob(flux_tunable_backend, pubs, InputType.INPUT_STREAM)

        assert isinstance(job.programs, list)
        assert len(job.programs) == 1
        assert isinstance(job.programs[0], Program)
        assert job._chunk_layout == [[0, 1, 2]]

    def test_multiple_programs_stored_as_list(self, flux_tunable_backend):
        from qiskit_qm_provider.job.qm_sampler_job import QMSamplerJob

        flux_tunable_backend.set_options(max_circuits=2)
        pubs = _sampler_pubs(flux_tunable_backend, 5)
        job = QMSamplerJob(flux_tunable_backend, pubs, InputType.INPUT_STREAM)

        assert isinstance(job.programs, list)
        assert len(job.programs) == 3  # ceil(5/2) = 3 chunks
        assert all(isinstance(p, Program) for p in job.programs)

    def test_chunk_layout_is_correct_partition(self, flux_tunable_backend):
        from qiskit_qm_provider.job.qm_sampler_job import QMSamplerJob

        flux_tunable_backend.set_options(max_circuits=3)
        pubs = _sampler_pubs(flux_tunable_backend, 7)
        job = QMSamplerJob(flux_tunable_backend, pubs, InputType.INPUT_STREAM)

        flat = [g for chunk in job._chunk_layout for g in chunk]
        assert flat == list(range(7))
        assert job._chunk_layout == [[0, 1, 2], [3, 4, 5], [6]]

    def test_locator_maps_all_pubs(self, flux_tunable_backend):
        from qiskit_qm_provider.job.qm_sampler_job import QMSamplerJob

        flux_tunable_backend.set_options(max_circuits=3)
        pubs = _sampler_pubs(flux_tunable_backend, 7)
        job = QMSamplerJob(flux_tunable_backend, pubs, InputType.INPUT_STREAM)

        assert sorted(job._locator.keys()) == list(range(7))
        # Global 3 is the first in chunk 1 → local index 0
        assert job._locator[3] == (1, 0)
        assert job._locator[6] == (2, 0)

    def test_locator_local_index_matches_stream_key_semantics(self, flux_tunable_backend):
        """The local index in the locator is the index used for creg stream keys."""
        from qiskit_qm_provider.job.qm_sampler_job import QMSamplerJob

        flux_tunable_backend.set_options(max_circuits=2)
        pubs = _sampler_pubs(flux_tunable_backend, 4)
        job = QMSamplerJob(flux_tunable_backend, pubs, InputType.INPUT_STREAM)

        # For each global pub, verify local index is its position within its chunk.
        for g, (chunk_idx, local_idx) in job._locator.items():
            assert job._chunk_layout[chunk_idx][local_idx] == g


# ---------------------------------------------------------------------------
# QMEstimatorJob construction (no submit)
# ---------------------------------------------------------------------------

class TestQMEstimatorJobConstruction:
    """Verify QMEstimatorJob chunking structure (chunk_layout, locator, programs list).

    ``plan_estimator_programs`` is patched to return dummy Programs so we can
    test chunking logic without the per-plan QUA compilation, which requires
    every plan to have fresh Parameter objects — a constraint that is only met
    in production by resetting the pool once and processing plans sequentially
    (which works when plans have unique circuit names and thus unique Parameter
    objects per obs var thanks to Parameter.__new__'s name-based singleton).
    """

    @staticmethod
    def _dummy_program():
        from qm.qua import program
        with program() as prog:
            pass
        return prog

    def _make_job(self, backend, count, max_circuits):
        from unittest.mock import patch
        from qiskit_qm_provider.job.qm_estimator_job import QMEstimatorJob
        from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorV2
        backend.set_options(max_circuits=max_circuits)
        pubs = _estimator_pubs(backend, count)
        switch_circ = QMEstimatorV2(
            backend, options={"input_type": InputType.INPUT_STREAM}
        )._switch_obs_circuit
        dummy = self._dummy_program()
        n_chunks = -(-count // max_circuits)  # ceil division
        fake_layout = [
            list(range(i, min(i + max_circuits, count)))
            for i in range(0, count, max_circuits)
        ]
        with patch(
            "qiskit_qm_provider.job.qm_estimator_job.plan_estimator_programs",
            return_value=([dummy] * n_chunks, fake_layout),
        ):
            return QMEstimatorJob(
                backend, pubs, InputType.INPUT_STREAM,
                switch_obs_circuit=switch_circ
            )

    def test_single_program_in_list(self, flux_tunable_backend):
        job = self._make_job(flux_tunable_backend, count=2, max_circuits=10)
        assert isinstance(job.programs, list)
        assert len(job.programs) == 1
        assert isinstance(job.programs[0], Program)
        assert job._chunk_layout == [[0, 1]]

    def test_multiple_programs_stored_as_list(self, flux_tunable_backend):
        job = self._make_job(flux_tunable_backend, count=5, max_circuits=2)
        assert isinstance(job.programs, list)
        assert len(job.programs) == 3
        assert all(isinstance(p, Program) for p in job.programs)

    def test_chunk_layout_is_correct_partition(self, flux_tunable_backend):
        job = self._make_job(flux_tunable_backend, count=5, max_circuits=2)
        flat = [g for chunk in job._chunk_layout for g in chunk]
        assert flat == list(range(5))
        assert job._chunk_layout == [[0, 1], [2, 3], [4]]

    def test_locator_maps_all_pubs(self, flux_tunable_backend):
        job = self._make_job(flux_tunable_backend, count=5, max_circuits=2)
        assert sorted(job._locator.keys()) == list(range(5))
        assert job._locator[2] == (1, 0)
        assert job._locator[4] == (2, 0)

    def test_locator_local_index_matches_chunk_position(self, flux_tunable_backend):
        job = self._make_job(flux_tunable_backend, count=5, max_circuits=2)
        for g, (chunk_idx, local_idx) in job._locator.items():
            assert job._chunk_layout[chunk_idx][local_idx] == g


# ---------------------------------------------------------------------------
# max_circuits as a backend option
# ---------------------------------------------------------------------------

class TestMaxCircuitsBackendOption:

    def test_default_is_30(self, flux_tunable_backend):
        assert flux_tunable_backend.options.max_circuits == 30

    def test_set_options_updates_value(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=5)
        assert flux_tunable_backend.options.max_circuits == 5

    def test_property_reads_from_options(self, flux_tunable_backend):
        flux_tunable_backend.set_options(max_circuits=7)
        assert flux_tunable_backend.max_circuits == 7


    def test_invalid_max_circuits_raises(self):
        with pytest.raises(ValueError, match="max_circuits must be a positive integer"):
            compute_chunk_layout(50, max_circuits=0)

    def test_max_circuits_one_splits_each_circuit(self):
        layout = compute_chunk_layout(4, max_circuits=1)
        assert layout == [[0], [1], [2], [3]]


    def test_sampler_job_respects_backend_max_circuits(self, flux_tunable_backend):
        from qiskit_qm_provider.job.qm_sampler_job import QMSamplerJob

        flux_tunable_backend.set_options(max_circuits=2)
        pubs = _sampler_pubs(flux_tunable_backend, 6)
        job = QMSamplerJob(flux_tunable_backend, pubs, InputType.INPUT_STREAM)
        assert len(job._chunk_layout) == 3  # 6 / 2 = 3 chunks

