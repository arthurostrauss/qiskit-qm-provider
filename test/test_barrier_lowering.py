"""Barriers through transpilation (optimization levels 0-3) and qm-qasm lowering to QUA ``align``.

Each circuit is transpiled against two backends -- a config-valid three-qubit dummy
``FluxTunableQuam`` from :mod:`test.dummy_quam` (always available) and, when
``QUAM_STATE_PATH`` is set, a module-owned backend over the
real machine (``quam_state_backend``) -- compiled with
``quantum_circuit_to_qua`` and rendered with ``generate_qua_script``. The checks follow
every barrier to the physical qubits the transpiler leaves it on (layout may permute
them at higher optimization levels) and expect qm-qasm to emit an ``align`` over exactly
the QUA elements of those qubits (``backend.qubit_mapping``), at the right point of the
program.

qm-qasm's align tracker skips an align whose elements are already in sync, so a
barrier immediately following another one on the same qubits emits no second align
(see ``test_back_to_back_barriers_emit_a_single_align``).
"""

import re

import pytest
from qiskit import transpile
from qiskit.circuit import ControlFlowOp, QuantumCircuit
from qiskit.circuit.classical import expr

from ._in_memory_quam import build_backend

qm = pytest.importorskip("qm")
from qm import generate_qua_script  # noqa: E402

OPTIMIZATION_LEVELS = [0, 1, 2, 3]
_ALIGN = re.compile(r"align\((?P<elements>[^)]*)\)")
# QM script generation can use either Python string delimiter.
_ELEMENT = re.compile(r'''["']([^"']+)["']''')


@pytest.fixture(scope="module")
def in_memory_backend(tmp_path_factory):
    # Patch QUAM_STATE_PATH only while building: leaving it patched would make the
    # session-scoped flux_tunable_backend load the temporary state instead of the real one.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("QUAM_STATE_PATH", str(tmp_path_factory.mktemp("quam_state")))
        return build_backend(3, conditional_plays=("x180",))


@pytest.fixture(scope="module")
def quam_state_backend(quam_machine, flux_tunable_backend):
    """A backend over the QuAM-state machine that this module owns.

    Built fresh (after ``flux_tunable_backend`` has seeded the machine's macros) rather
    than reusing the session-scoped fixture, so registering conditional plays here does
    not change the Target other test modules see.
    """
    from qiskit_qm_provider import QMProvider
    from qiskit_qm_provider.backend.flux_tunable_transmon_backend import FluxTunableTransmonBackend

    return QMProvider().get_backend(quam_machine, FluxTunableTransmonBackend)


@pytest.fixture(params=["in_memory", "quam_state"])
def backend(request):
    """The in-memory backend, or the QuAM-state backend (skipped when QUAM_STATE_PATH is unset)."""
    if request.param == "in_memory":
        return request.getfixturevalue("in_memory_backend")
    return request.getfixturevalue("quam_state_backend")


def _barrier_qubit_sets(circuit: QuantumCircuit, qubit_map=None):
    """Physical qubits of every barrier in ``circuit``, including inside control flow, in program order."""
    if qubit_map is None:
        qubit_map = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    sets = []
    for instruction in circuit.data:
        qargs = [qubit_map[qubit] for qubit in instruction.qubits]
        if instruction.operation.name == "barrier":
            sets.append(frozenset(qargs))
        elif isinstance(instruction.operation, ControlFlowOp):
            for block in instruction.operation.blocks:
                sets.extend(_barrier_qubit_sets(block, dict(zip(block.qubits, qargs))))
    return sets


def _elements(backend, physical_qubits) -> frozenset:
    return frozenset(element for qubit in physical_qubits for element in backend.qubit_mapping[qubit])


def _align_elements(line: str):
    """The element set of a QUA ``align(...)`` line, or ``None`` for any other line."""
    match = _ALIGN.search(line)
    return frozenset(_ELEMENT.findall(match["elements"])) if match else None


def _lower(backend, qc: QuantumCircuit, level: int):
    """Transpile at ``level``, compile to QUA, and return ``(transpiled, stripped QUA lines)``."""
    transpiled = transpile(qc, backend, optimization_level=level, seed_transpiler=7)
    script = generate_qua_script(backend.quantum_circuit_to_qua(transpiled).qua_program)
    return transpiled, [line.strip() for line in script.splitlines()]


def _index_of_align(lines, elements) -> int:
    indices = [index for index, line in enumerate(lines) if _align_elements(line) == elements]
    assert indices, f"no align over exactly {sorted(elements)} in:\n" + "\n".join(lines)
    return indices[0]


def _assert_every_barrier_aligns_its_qubits(backend, qc, transpiled, lines):
    barrier_sets = _barrier_qubit_sets(transpiled)
    assert len(barrier_sets) == len(_barrier_qubit_sets(qc)), "transpilation dropped or added a barrier"
    for qubits in barrier_sets:
        _index_of_align(lines, _elements(backend, qubits))


def _play_indices(backend, lines, physical_qubit):
    """Indices of the QUA ``play`` lines on any element of ``physical_qubit``."""
    elements = set(backend.qubit_mapping[physical_qubit])
    return [
        index
        for index, line in enumerate(lines)
        if line.startswith("play(") and elements & set(_ELEMENT.findall(line))
    ]


@pytest.mark.parametrize("level", OPTIMIZATION_LEVELS)
def test_barrier_between_gates_aligns_only_its_qubits(backend, level):
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.sx(1)
    qc.barrier(0, 1)
    qc.x(0)
    qc.x(1)
    qc.x(2)
    qc.measure([0, 1, 2], [0, 1, 2])

    transpiled, lines = _lower(backend, qc, level)

    _assert_every_barrier_aligns_its_qubits(backend, qc, transpiled, lines)
    [barrier_qubits] = _barrier_qubit_sets(transpiled)
    align_index = _index_of_align(lines, _elements(backend, barrier_qubits))
    for qubit in barrier_qubits:
        before, after = _play_indices(backend, lines, qubit)
        assert before < align_index < after, f"align is not between the two plays on q{qubit}"


@pytest.mark.parametrize("level", OPTIMIZATION_LEVELS)
def test_barrier_on_a_subset_skips_the_other_qubits(backend, level):
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.x(1)
    qc.x(2)
    qc.barrier(0, 2)
    qc.x(0)
    qc.x(2)
    qc.measure([0, 2], [0, 2])

    transpiled, lines = _lower(backend, qc, level)

    _assert_every_barrier_aligns_its_qubits(backend, qc, transpiled, lines)
    [barrier_qubits] = _barrier_qubit_sets(transpiled)
    assert len(barrier_qubits) == 2
    # _assert_every_barrier_aligns_its_qubits matched an align over exactly these two
    # qubits' elements, so the third qubit is not synchronized with them.


@pytest.mark.parametrize("level", OPTIMIZATION_LEVELS)
def test_barrier_at_the_start_aligns_every_qubit(backend, level):
    qc = QuantumCircuit(3, 1)
    qc.barrier()
    qc.x(0)
    qc.measure(0, 0)

    transpiled, lines = _lower(backend, qc, level)

    _assert_every_barrier_aligns_its_qubits(backend, qc, transpiled, lines)
    [barrier_qubits] = _barrier_qubit_sets(transpiled)
    assert len(barrier_qubits) == 3
    align_index = _index_of_align(lines, _elements(backend, barrier_qubits))
    assert align_index < min(index for index, line in enumerate(lines) if line.startswith("play("))


@pytest.mark.parametrize("level", OPTIMIZATION_LEVELS)
def test_back_to_back_barriers_emit_a_single_align(backend, level):
    qc = QuantumCircuit(3, 1)
    qc.x(0)
    qc.x(1)
    qc.barrier(0, 1)
    qc.barrier(0, 1)
    qc.x(0)
    qc.measure(0, 0)

    transpiled, lines = _lower(backend, qc, level)

    _assert_every_barrier_aligns_its_qubits(backend, qc, transpiled, lines)
    barrier_qubits = _barrier_qubit_sets(transpiled)[0]
    elements = _elements(backend, barrier_qubits)
    assert sum(_align_elements(line) == elements for line in lines) == 1


@pytest.mark.parametrize("level", OPTIMIZATION_LEVELS)
def test_measure_all_barrier_aligns_before_every_measurement(backend, level):
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.sx(2)
    qc.measure_all()

    transpiled, lines = _lower(backend, qc, level)

    _assert_every_barrier_aligns_its_qubits(backend, qc, transpiled, lines)
    [barrier_qubits] = _barrier_qubit_sets(transpiled)
    assert len(barrier_qubits) == 3
    align_index = _index_of_align(lines, _elements(backend, barrier_qubits))
    measure_indices = [index for index, line in enumerate(lines) if line.startswith("measure(")]
    assert len(measure_indices) == 3
    assert align_index < min(measure_indices)


@pytest.mark.parametrize("level", OPTIMIZATION_LEVELS)
def test_barrier_inside_if_test_aligns_inside_the_branch(backend, level):
    qc = QuantumCircuit(3, 1)
    qc.x(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
        qc.barrier(1, 2)
        qc.x(2)

    transpiled, lines = _lower(backend, qc, level)

    _assert_every_barrier_aligns_its_qubits(backend, qc, transpiled, lines)
    [barrier_qubits] = _barrier_qubit_sets(transpiled)
    align_index = _index_of_align(lines, _elements(backend, barrier_qubits))
    if_index = next(index for index, line in enumerate(lines) if line.startswith("with if_("))
    assert if_index < align_index


@pytest.mark.parametrize("level", OPTIMIZATION_LEVELS)
def test_barrier_before_a_conditional_play(backend, level):
    if "qm_conditional_play_x180" not in backend.target.operation_names:
        try:
            backend.register_conditional_play("x180")
        except ValueError as error:
            pytest.skip(f"cannot register a conditional x180 play on this machine: {error}")
    qc = QuantumCircuit(3, 1)
    qc.x(1)
    qc.measure(1, 0)
    qc.barrier(0, 1)
    qc.conditional_play("x180", expr.lift(qc.clbits[0]), 0)

    transpiled, lines = _lower(backend, qc, level)

    _assert_every_barrier_aligns_its_qubits(backend, qc, transpiled, lines)
    [barrier_qubits] = _barrier_qubit_sets(transpiled)
    align_index = _index_of_align(lines, _elements(backend, barrier_qubits))
    conditional_index = next(index for index, line in enumerate(lines) if "condition=" in line)
    measure_index = next(index for index, line in enumerate(lines) if line.startswith("measure("))
    assert measure_index < align_index < conditional_index


@pytest.mark.parametrize("level", OPTIMIZATION_LEVELS)
def test_barrier_after_a_routed_two_qubit_gate(backend, level):
    """``cx`` between non-neighbours: level 0 routes it with swaps, higher levels relayout
    instead. Either way the barrier follows its qubits to their final physical positions."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 2)
    qc.barrier(0, 2)
    qc.x(0)
    qc.x(2)
    qc.measure([0, 1, 2], [0, 1, 2])

    transpiled, lines = _lower(backend, qc, level)

    assert "cz" in transpiled.count_ops()
    _assert_every_barrier_aligns_its_qubits(backend, qc, transpiled, lines)
