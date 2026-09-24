"""Tests for the host-side mirror of QUA's ``Random`` (``qiskit_qm_provider.random``).

Two kinds of ground truth are used:

- an independent reference LCG (numpy int32 wrap-around and a float ``rand_int``
  formula), cross-checked against hardware;
- raw values recorded from the IQCC "arbel" controller.

Everything except the opt-in hardware tests is fixture-free.
"""

import os

import numpy as np
import pytest

from qiskit_qm_provider.fixed_point import FixedPoint
from qiskit_qm_provider.random import BoxMullerTables, Random, box_muller_pair, rand_gauss_box_muller
from qiskit_qm_provider.random.lcg import STATE_MASK

# --- Reference LCG (independent formulation, matches hardware) ----------------------

_A, _C, _M = 137939405, 12345, 2**28


def _ref_lcg(seed):
    seed = np.int32(seed)
    with np.errstate(over="ignore"):
        return (np.int32(_A) * seed + np.int32(_C)) % _M


def _ref_lcg_int(seed, upper_bound):
    seed = _ref_lcg(seed)
    return np.int32(seed / 2**28 * upper_bound), seed


def _ref_lcg_fixed_point(seed):
    seed = _ref_lcg(seed)
    return (seed & 0x0FFFFFFF) / float(2**28), seed


_SEEDS = [0, 1, 42, 12345, STATE_MASK, 2**28, 2**31 - 1, -1, -(2**31), 987654321]


# --- Random ---------------------------------------------------------------------------


class TestRandomMatchesReference:
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_rand_fixed_sequence(self, seed):
        rng = Random(seed)
        ref_seed = seed
        for _ in range(200):
            expected, ref_seed = _ref_lcg_fixed_point(ref_seed)
            value = rng.rand_fixed()
            assert isinstance(value, FixedPoint)
            assert float(value) == expected
            assert value.to_unsafe_int() == rng.state == int(ref_seed)

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("max_int", [1, 2, 7, 24, 1000, 2**31 - 1])
    def test_rand_int_sequence(self, seed, max_int):
        rng = Random(seed)
        ref_seed = seed
        for _ in range(100):
            expected, ref_seed = _ref_lcg_int(ref_seed, max_int)
            assert rng.rand_int(max_int) == int(expected)

    def test_matches_recorded_hardware_draws(self):
        # Raw 4.28 values of successive rand_fixed() draws from Random(424242) on IQCC
        # "arbel" (2026-09-24).
        recorded = [222789443, 64271328, 193927833, 116563390, 263368543, 43005516, 146626325, 227961098]
        assert [v.to_unsafe_int() for v in Random(424242).rand_fixed_sequence(8)] == recorded

    def test_many_random_seeds(self):
        seeds = np.random.default_rng(0).integers(-(2**31), 2**31, 500)
        for seed in seeds:
            expected, _ = _ref_lcg_fixed_point(seed)
            assert float(Random(int(seed)).rand_fixed()) == expected

    def test_seed_int32_wrap(self):
        assert Random(2**32 + 5).rand_int_sequence(10, 20) == Random(5).rand_int_sequence(10, 20)
        assert Random(2**31).initial_seed == -(2**31)

    def test_only_seed_mod_2_28_matters(self):
        assert Random(7 + 3 * 2**28).rand_int_sequence(100, 20) == Random(7).rand_int_sequence(100, 20)


class TestRandomApi:
    def test_rand_int_range(self):
        values = Random(3).rand_int_sequence(5, 1000)
        assert set(values) == set(range(5))

    def test_rand_int_rejects_non_positive(self):
        with pytest.raises(ValueError):
            Random(3).rand_int(0)

    def test_rand_fixed_range(self):
        values = [float(v) for v in Random(3).rand_fixed_sequence(1000)]
        assert all(0.0 <= v < 1.0 for v in values)

    def test_unseeded_exposes_seed(self):
        rng = Random()
        assert 0 <= rng.initial_seed < STATE_MASK
        assert Random(rng.initial_seed).rand_int_sequence(10, 5) == rng.rand_int_sequence(10, 5)

    def test_draws_and_set_seed(self):
        rng = Random(11)
        rng.rand_fixed()
        rng.rand_int(4)
        assert rng.draws == 2
        rng.set_seed(11)
        assert rng.draws == 0
        assert rng.state == 11
        assert rng.rand_int_sequence(100, 5) == Random(11).rand_int_sequence(100, 5)

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 17, 1000])
    def test_skip_matches_sequential_draws(self, n):
        skipped, stepped = Random(-99), Random(-99)
        skipped.skip(n)
        stepped.rand_fixed_sequence(n)
        assert skipped.state == stepped.state
        assert skipped.draws == stepped.draws == n

    def test_skip_rejects_negative(self):
        with pytest.raises(ValueError):
            Random(1).skip(-1)

    def test_copy_is_independent(self):
        rng = Random(5)
        rng.rand_fixed()
        clone = rng.copy()
        assert clone.state == rng.state and clone.draws == rng.draws
        clone.rand_fixed()
        assert clone.state != rng.state


# --- Box-Muller -----------------------------------------------------------------------


class TestBoxMuller:
    def test_tables_validation(self):
        for bad in (0, 2, 3, 100, 2**15):
            with pytest.raises(ValueError):
                BoxMullerTables(bad)
        tables = BoxMullerTables.get()
        assert tables.n_lookup == 512
        assert tables.u1_shift == 19
        assert BoxMullerTables.get() is tables

    def test_one_draw_per_dimension(self):
        rng = Random(1234)
        for _ in range(16):
            box_muller_pair(rng, [0.1, -0.3, 0.05, 0.4, -0.2], [0.2, 0.05, 0.5, 0.1, 0.3])
        assert rng.draws == 16 * 5

    def test_matches_recorded_hardware_samples(self):
        # Raw 4.28 values streamed from IQCC "arbel" (2026-09-24): a fresh Random(424242)
        # feeding rand_gauss_box_muller with mean=[0.1, -0.2], std=[0.3, 0.05], saved as
        # z1[0], z2[0], z1[1], z2[1] per call.
        recorded = [
            -6781520, 63040301, -32730617, -45006635,
            7182138, -35219267, -41740767, -41139475,
            20547922, 41530123, -38426940, -74263023,
        ]
        rng = Random(424242)
        host = []
        for _ in range(3):
            z1, z2 = box_muller_pair(rng, [0.1, -0.2], [0.3, 0.05])
            for a, b in zip(z1, z2):
                host += [a.to_unsafe_int(), b.to_unsafe_int()]
        assert host == recorded

    def test_outputs_are_fixed_point(self):
        z1, z2 = box_muller_pair(Random(1), [FixedPoint(0.0)], [FixedPoint(1.0)])
        assert isinstance(z1[0], FixedPoint) and isinstance(z2[0], FixedPoint)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            box_muller_pair(Random(1), [0.0, 0.0], [1.0])

    def test_statistics(self):
        rng = Random(2024)
        samples = []
        for _ in range(4000):
            z1, z2 = box_muller_pair(rng, [0.0], [1.0])
            samples += [float(z1[0]), float(z2[0])]
        assert abs(np.mean(samples)) < 0.05
        assert abs(np.std(samples) - 1.0) < 0.05


# --- QUA side (no hardware) -----------------------------------------------------------


class TestQuaGeneration:
    def test_requires_program_scope(self):
        with pytest.raises(RuntimeError):
            Random(1).declare_qua()
        with pytest.raises(RuntimeError):
            BoxMullerTables.get().declare_qua()
        with pytest.raises(RuntimeError):
            rand_gauss_box_muller(None, None, None, None, None)

    def test_declare_qua_uses_current_state(self):
        from qm import generate_qua_script
        from qm.qua import assign, declare, program

        rng = Random(5)
        rng.rand_fixed()
        with program() as prog:
            qua_rng = rng.declare_qua()
            x = declare(int)
            assign(x, qua_rng.rand_int(10))
        script = generate_qua_script(prog)
        assert f"declare(int, value={rng.state})" in script
        assert ".rand_int(10)" in script

    def test_gaussian_macro_script(self):
        from qm import generate_qua_script
        from qm.qua import declare, fixed, for_, program

        rng = Random(7)
        with program() as prog:
            qua_rng = rng.declare_qua()
            mean = declare(fixed, value=[0.0, 0.1])
            std = declare(fixed, value=[0.5, 1.0])
            z1 = declare(fixed, size=2)
            z2 = declare(fixed, size=2)
            tables = BoxMullerTables.get().declare_qua()
            b = declare(int)
            with for_(b, 0, b < 4, b + 2):
                rand_gauss_box_muller(qua_rng, mean, std, z1, z2, tables_qua=tables)
        script = generate_qua_script(prog)
        assert script.count(".rand_fixed()") == 1
        assert ">>19" in script
        assert script.count("declare(fixed, value=[1.0, ") == 1  # cos table declared once


# --- Opt-in hardware checks -----------------------------------------------------------

_HW_SEED, _HW_N = 424242, 64
_HW_MEAN, _HW_STD = [0.1, -0.2], [0.3, 0.05]


def _hardware_program():
    from qm.qua import assign, declare, declare_stream, fixed, for_, program, save, stream_processing

    with program() as prog:
        qua_rng = Random(_HW_SEED).declare_qua()
        i = declare(int)
        k = declare(int)
        u = declare(fixed)
        mean = declare(fixed, value=_HW_MEAN)
        std = declare(fixed, value=_HW_STD)
        z1 = declare(fixed, size=len(_HW_MEAN))
        z2 = declare(fixed, size=len(_HW_MEAN))
        int_stream, fixed_stream, gauss_stream = declare_stream(), declare_stream(), declare_stream()
        with for_(i, 0, i < _HW_N, i + 1):
            assign(k, qua_rng.rand_int(1000))
            save(k, int_stream)
            assign(u, qua_rng.rand_fixed())
            save(u, fixed_stream)
            rand_gauss_box_muller(qua_rng, mean, std, z1, z2)
            with for_(k, 0, k < len(_HW_MEAN), k + 1):
                save(z1[k], gauss_stream)
                save(z2[k], gauss_stream)
        with stream_processing():
            int_stream.save_all("ints")
            fixed_stream.save_all("fixed")
            gauss_stream.save_all("gauss")
    return prog


def _host_expectation():
    rng = Random(_HW_SEED)
    ints, fixeds, gauss = [], [], []
    for _ in range(_HW_N):
        ints.append(rng.rand_int(1000))
        fixeds.append(float(rng.rand_fixed()))
        z1, z2 = box_muller_pair(rng, _HW_MEAN, _HW_STD)
        for a, b in zip(z1, z2):
            gauss += [float(a), float(b)]
    return {"ints": ints, "fixed": fixeds, "gauss": gauss}


def _fetch(result_handles, name):
    data = np.asarray(result_handles.get(name).fetch_all())
    if data.dtype.names and "value" in data.dtype.names:
        data = data["value"]
    return data.ravel()


def _run_and_compare(machine):
    """Execute the draw program on ``machine`` and compare every stream with the host."""
    qmm = machine.connect()
    qm = qmm.open_qm(machine.generate_config(), close_other_machines=True)
    job = qm.execute(_hardware_program())
    job.result_handles.wait_for_all_values()
    expected = _host_expectation()
    mismatches = {}
    for name, values in expected.items():
        measured = _fetch(job.result_handles, name)
        if measured.shape != (len(values),) or not np.array_equal(measured, values):
            mismatches[name] = (measured[:6].tolist(), values[:6])
    assert not mismatches, f"controller vs host (first values): {mismatches}"


@pytest.mark.skipif(
    not os.environ.get("QM_RANDOM_HARDWARE_TEST"),
    reason="set QM_RANDOM_HARDWARE_TEST=1 (and QUAM_STATE_PATH) to compare against the controller",
)
def test_hardware_matches_python(quam_machine):
    """Stream controller draws and compare them bit-for-bit with the host mirror."""
    _run_and_compare(quam_machine)


@pytest.mark.skipif(
    not os.environ.get("QM_RANDOM_IQCC_BACKEND"),
    reason="set QM_RANDOM_IQCC_BACKEND=<name> (e.g. arbel) to compare against an IQCC machine",
)
def test_iqcc_hardware_matches_python(tmp_path):
    """Same check on an IQCC machine, loading its latest state into a temporary folder."""
    from qiskit_qm_provider.providers.iqcc_cloud_provider import IQCCProvider

    machine = IQCCProvider().get_machine(os.environ["QM_RANDOM_IQCC_BACKEND"], quam_state_folder_path=str(tmp_path))
    _run_and_compare(machine)
