"""``ensure_native_operations`` against a Target seeded by ``add_basic_macros``.

Builds a real two-qubit ``FluxTunableQuam`` in memory with quam-builder (no QuAM state
folder or hardware), so the Target holds the same macro-derived operations as a
production backend: ``x``/``y``/``sx``/``sxdg``/``sy``/``sydg``/``rz``, ``measure``,
``reset``, ``delay``, ``id`` and control flow, plus registered conditional plays.
"""

import math

import pytest
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical import expr, types
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_qm_provider import add_basic_macros
from qiskit_qm_provider.backend.backend_utils import (
    ensure_circuit_physical,
    ensure_native_operations,
    validate_circuits,
)
from qiskit_qm_provider.backend.qm_backend import QMBackend

quam_builder = pytest.importorskip("quam_builder")
from quam.components.ports import FEMPortsContainer  # noqa: E402
from quam.components.pulses import SquarePulse  # noqa: E402
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam  # noqa: E402
from quam_builder.builder.superconducting.modify_quam import add_qubit  # noqa: E402

# Canonical pulse names add_basic_macros expects (quam-builder's defaults use other names).
_CANONICAL_PULSES = [
    ("x180", 0.2, 0.0),
    ("y180", 0.2, math.pi / 2),
    ("x90", 0.1, 0.0),
    ("-x90", 0.1, math.pi),
    ("y90", 0.1, math.pi / 2),
    ("-y90", 0.1, -math.pi / 2),
]


def _machine():
    machine = FluxTunableQuam(ports=FEMPortsContainer())
    for index in range(2):
        add_qubit(
            machine,
            f"q{index}",
            {
                "xy": {"opx_output": f"#/ports/mw_outputs/con1/1/{index + 1}"},
                "rr": {"opx_output": "#/ports/mw_outputs/con1/2/1", "opx_input": "#/ports/mw_inputs/con1/2/1"},
                "z": {"opx_output": f"#/ports/analog_outputs/con1/3/{index + 1}"},
            },
        )
    for qubit in machine.active_qubits:
        for name, amplitude, axis_angle in _CANONICAL_PULSES:
            qubit.xy.operations[name] = SquarePulse(length=40, amplitude=amplitude, axis_angle=axis_angle)
    return machine


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAM_STATE_PATH", str(tmp_path))
    backend = QMBackend(_machine())
    add_basic_macros(backend)
    backend.register_conditional_play("x180", "-y90")
    return backend


def _check(qc, backend):
    ensure_circuit_physical(qc, backend.num_qubits)
    ensure_native_operations(qc, backend.target)


def test_run_path_circuit_is_native(backend):
    """A transpiled dynamic circuit, after the reset ``validate_circuits`` adds for
    ``backend.run``/the Sampler, passes the native-operation check."""
    qc = QuantumCircuit(2, 2)
    flag = qc.add_input("flag", types.Bool())
    seen = qc.add_var("seen", False)
    qc.h(0)
    qc.sdg(1)
    qc.delay(100, 0)
    qc.barrier()
    qc.measure(1, 0)
    qc.store(seen, expr.lift(qc.clbits[0]))
    qc.conditional_play("x180", expr.logic_and(flag, qc.clbits[0]), 0)
    with qc.if_test(expr.lift(qc.clbits[0])):
        qc.x(1)
    qc.measure([0, 1], [0, 1])

    transpiled = transpile(qc, backend, optimization_level=2, seed_transpiler=0)
    [validated] = validate_circuits(transpiled)

    _check(validated, backend)
    qasm = backend.qasm3_exporter.dumps(validated)
    assert any(line in qasm for line in ("barrier $0, $1;", "barrier $1, $0;"))
    assert "seen = c[0];" in qasm


def test_estimator_prepared_circuit_is_native(backend):
    """The Estimator's composed ``switch`` basis-change block stays native."""
    from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorV2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.sx(1)
    transpiled = transpile(qc, backend, optimization_level=1, seed_transpiler=0)
    observables = SparsePauliOp(["XZ", "YY"]).apply_layout(transpiled.layout)
    pub = EstimatorPub.coerce((transpiled, observables), precision=0.01)

    [prepared] = QMEstimatorV2(backend).validate_estimator_pubs([pub])

    _check(prepared.circuit, backend)


def test_non_native_operation_is_named_in_the_error(backend):
    qc = QuantumCircuit(2)
    qc.x(0)
    transpiled = transpile(qc, backend, optimization_level=0, seed_transpiler=0)
    with transpiled.if_test(expr.lift(True)):
        transpiled.h(1)

    with pytest.raises(ValueError, match=r"\['h'\] that the given Target does not define"):
        ensure_native_operations(transpiled, backend.target)


def test_qubit_mapping_uses_quam_element_names(backend):
    """The base backend maps each physical qubit to its channels' QUA element names."""
    assert backend.qubit_mapping == {
        index: (f"q{index}.xy", f"q{index}.resonator", f"q{index}.z") for index in range(2)
    }


def test_barrier_compiles_to_an_align_over_its_qubits(backend):
    """quantum_circuit_to_qua keeps barriers, and qm-qasm lowers one to a QUA align over
    every element of the barrier's qubits."""
    from qm import generate_qua_script

    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.barrier()
    qc.x(1)
    qc.measure([0, 1], [0, 1])
    transpiled = transpile(qc, backend, optimization_level=0, initial_layout=[0, 1])

    script = generate_qua_script(backend.quantum_circuit_to_qua(transpiled).qua_program)

    lines = [line.strip() for line in script.splitlines()]
    first, second = lines.index("play('x180', 'q0.xy')"), lines.index("play('x180', 'q1.xy')")
    assert "align('q0.resonator', 'q0.xy', 'q0.z', 'q1.resonator', 'q1.xy', 'q1.z')" in lines[first:second]


def test_flux_tunable_qubit_mapping_extends_the_base_mapping(tmp_path, monkeypatch):
    """FluxTunableTransmonBackend reuses QMBackend.qubit_mapping and only adds TWPA
    pumps; this machine has none, so the two mappings coincide."""
    from qiskit_qm_provider.backend.flux_tunable_transmon_backend import FluxTunableTransmonBackend

    monkeypatch.setenv("QUAM_STATE_PATH", str(tmp_path))
    machine = _machine()

    assert FluxTunableTransmonBackend(machine).qubit_mapping == QMBackend(machine).qubit_mapping
