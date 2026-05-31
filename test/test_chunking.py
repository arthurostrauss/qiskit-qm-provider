"""Tests for batch chunking of circuits into multiple QUA programs.

These tests exercise the pure index arithmetic behind ``backend.run``'s
splitting of large circuit batches (``max_circuits``). They are deliberately
free of any QuAM/hardware fixtures so they run in any environment.
"""

import pytest

from qiskit import QuantumCircuit
from qm import Program

from qiskit_qm_provider.job.qua_programs import (
    compute_chunk_layout,
    get_run_program,
    plan_run_programs,
)


def _flatten(layout):
    return [g for chunk in layout for g in chunk]


class TestComputeChunkLayout:
    def test_single_program_when_under_limit(self):
        assert compute_chunk_layout(5, max_circuits=30) == [[0, 1, 2, 3, 4]]

    def test_single_program_when_exactly_at_limit(self):
        layout = compute_chunk_layout(30, max_circuits=30)
        assert len(layout) == 1
        assert layout[0] == list(range(30))

    def test_split_just_over_limit(self):
        # 31 circuits, limit 30 -> [0..29], [30]
        layout = compute_chunk_layout(31, max_circuits=30)
        assert layout == [list(range(30)), [30]]

    def test_split_consecutive_groups(self):
        # 70 circuits, limit 30 -> [0..29], [30..59], [60..69]
        layout = compute_chunk_layout(70, max_circuits=30)
        assert len(layout) == 3
        assert layout[0] == list(range(0, 30))
        assert layout[1] == list(range(30, 60))
        assert layout[2] == list(range(60, 70))

    def test_max_circuits_none_never_splits(self):
        layout = compute_chunk_layout(100, max_circuits=None)
        assert layout == [list(range(100))]

    def test_conflicting_calibrations_one_per_program(self):
        layout = compute_chunk_layout(
            4, max_circuits=30, conflicting_calibrations=True
        )
        assert layout == [[0], [1], [2], [3]]

    def test_conflicting_takes_priority_over_size(self):
        # Conflicts force one-per-program even past the size limit; every chunk
        # is then trivially within max_circuits.
        layout = compute_chunk_layout(
            50, max_circuits=30, conflicting_calibrations=True
        )
        assert layout == [[i] for i in range(50)]
        assert all(len(chunk) <= 30 for chunk in layout)

    @pytest.mark.parametrize("n", [1, 5, 29, 30, 31, 60, 61, 70, 100])
    def test_layout_is_a_partition_in_order(self, n):
        # Every global index appears exactly once, in ascending order.
        layout = compute_chunk_layout(n, max_circuits=30)
        assert _flatten(layout) == list(range(n))
        assert all(len(chunk) <= 30 for chunk in layout)


class TestResultLocator:
    """The result-stitching locator: global index -> (chunk, local index)."""

    @staticmethod
    def _locator(layout):
        return {
            g: (c, l)
            for c, chunk in enumerate(layout)
            for l, g in enumerate(chunk)
        }

    def test_locator_single_program_is_identity(self):
        layout = compute_chunk_layout(40, max_circuits=None)
        locator = self._locator(layout)
        # Single program: chunk 0, local index == global index (back-compat key).
        for g in range(40):
            assert locator[g] == (0, g)

    def test_locator_maps_global_to_local_within_chunk(self):
        layout = compute_chunk_layout(70, max_circuits=30)
        locator = self._locator(layout)
        # Global 0 -> program 0, local 0
        assert locator[0] == (0, 0)
        # Global 30 -> program 1, local 0 (stream key would be "creg_0")
        assert locator[30] == (1, 0)
        # Global 59 -> program 1, local 29
        assert locator[59] == (1, 29)
        # Global 60 -> program 2, local 0
        assert locator[60] == (2, 0)
        assert locator[69] == (2, 9)

    def test_every_circuit_is_locatable_exactly_once(self):
        layout = compute_chunk_layout(61, max_circuits=30)
        locator = self._locator(layout)
        assert sorted(locator.keys()) == list(range(61))
        # Each (chunk, local) pair is unique.
        assert len(set(locator.values())) == 61


def _measure_circuit():
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    return qc


class TestPlanRunPrograms:
    """Build real QUA programs through the planner (requires a QuAM machine)."""

    def test_splits_into_multiple_programs(self, flux_tunable_backend):
        circuits = [_measure_circuit() for _ in range(7)]
        programs, layout = plan_run_programs(
            flux_tunable_backend, 100, circuits, max_circuits=3
        )
        assert layout == [[0, 1, 2], [3, 4, 5], [6]]
        assert len(programs) == 3
        assert all(isinstance(p, Program) for p in programs)

    def test_single_program_under_limit(self, flux_tunable_backend):
        circuits = [_measure_circuit() for _ in range(3)]
        programs, layout = plan_run_programs(
            flux_tunable_backend, 100, circuits, max_circuits=30
        )
        assert layout == [[0, 1, 2]]
        assert len(programs) == 1
        assert isinstance(programs[0], Program)

    def test_get_run_program_returns_bare_program_for_single_chunk(
        self, flux_tunable_backend
    ):
        # Backward-compat wrapper: a single program is returned bare (not a list).
        circuits = [_measure_circuit() for _ in range(40)]
        prog = get_run_program(flux_tunable_backend, 100, circuits)
        assert isinstance(prog, Program)
