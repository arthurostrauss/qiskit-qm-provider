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


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAM_STATE_PATH", str(tmp_path))
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

    backend = QMBackend(machine)
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
    qc.h(0)
    qc.sdg(1)
    qc.delay(100, 0)
    qc.measure(1, 0)
    qc.conditional_play("x180", expr.logic_and(flag, qc.clbits[0]), 0)
    with qc.if_test(expr.lift(qc.clbits[0])):
        qc.x(1)
    qc.measure([0, 1], [0, 1])

    transpiled = transpile(qc, backend, optimization_level=2, seed_transpiler=0)
    [validated] = validate_circuits(transpiled)

    _check(validated, backend)


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

    with pytest.raises(ValueError, match=r"'h' on qubit\(s\) \(1,\)"):
        ensure_native_operations(transpiled, backend.target)
