"""``ensure_native_operations`` against the Target of a realistic, config-valid QuAM.

Uses a two-qubit flux-tunable dummy QuAM built by :mod:`test.dummy_quam` (no QMM, QuAM
state folder or hardware), so the Target holds the operations quam-builder's default
macros produce on a production machine: ``x``/``y``/``sx``/``sy``/``rz``, ``measure``,
``reset``, ``delay``, ``id``, ``cz`` on the qubit pair, control flow, ``barrier``/``store``,
plus registered conditional plays.
"""

import pytest
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical import expr, types
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_qm_provider.backend.backend_utils import (
    ensure_circuit_physical,
    ensure_native_operations,
    validate_circuits,
)
from qiskit_qm_provider.backend.qm_backend import QMBackend

from ._in_memory_quam import build_backend, build_machine


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAM_STATE_PATH", str(tmp_path))
    return build_backend()


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


def test_flux_tunable_qubit_mapping_extends_the_base_mapping(tmp_path, monkeypatch):
    """FluxTunableTransmonBackend reuses QMBackend.qubit_mapping and only adds TWPA
    pumps; this machine has none, so the two mappings coincide."""
    from qiskit_qm_provider.backend.flux_tunable_transmon_backend import FluxTunableTransmonBackend

    monkeypatch.setenv("QUAM_STATE_PATH", str(tmp_path))
    machine = build_machine()

    assert FluxTunableTransmonBackend(machine).qubit_mapping == QMBackend(machine).qubit_mapping
