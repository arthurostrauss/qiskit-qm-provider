"""End-to-end correctness of estimator planning and post-processing on multi-qubit observables.

No hardware: the OPX is emulated exactly as the provider wires it. For each stream buffer the
plan's per-qubit observable codes select the basis rotation (code 1: ``h``, code 2: ``sdg; h``)
of the qubit measured into ``__c[j]``; outcomes are sampled from the exact statevector and
laid out with column ``j`` = ``__c[j]``. The planner, ``counts_from_estimator_stream``,
expectation values and result assembly then run unmodified (``estimator_pub_result``, shared with the job) and are compared with
``Statevector.expectation_value``.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.classical import types
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap

from qiskit_qm_provider.job.primitive_plans import EstimatorPlan
from qiskit_qm_provider.job.qm_estimator_job import estimator_pub_result
from qiskit_qm_provider.parameter_table import InputType, ParameterPool
from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions

PRECISION = 0.01  # 10_000 shots
TOL = 0.05


@pytest.fixture(autouse=True)
def reset_pool():
    ParameterPool.reset()
    yield
    ParameterPool.reset()


def _with_estimator_readout(circuit: QuantumCircuit, measured: list[int]) -> QuantumCircuit:
    """What estimator validation adds: one ``__obs_j`` input and one ``__c[j]`` bit per measured qubit."""
    qc = circuit.copy()
    qc.add_register(ClassicalRegister(len(measured), "__c"))
    for j in range(len(measured)):
        qc.add_input(f"__obs_{j}", types.Uint(4))
    if measured:
        qc.measure(measured, range(len(measured)))
    return qc


def _emulate_stream(plan, state_circuit: QuantumCircuit, measured: list[int], seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    buffers = [codes for group in plan.obs_codes for codes in group]
    raw = np.zeros((len(buffers), plan.shots, len(measured)), dtype=int)
    for b, codes in enumerate(buffers):
        rotated = state_circuit.copy()
        for j, code in enumerate(codes):
            if code == 2:
                rotated.sdg(measured[j])
            if code in (1, 2):
                rotated.h(measured[j])
        probs = Statevector(rotated).probabilities_dict(qargs=measured)
        keys = list(probs)
        draws = rng.choice(len(keys), size=plan.shots, p=[probs[k] for k in keys])
        for s, d in enumerate(draws):
            bits = keys[d][::-1]  # probabilities_dict keys are little-endian over qargs
            raw[b, s] = [int(c) for c in bits]
    return raw


def _estimate(plan) -> np.ndarray:
    raw = _emulate_stream(plan, _STATE[plan.pub.circuit.name], _MEAS[plan.pub.circuit.name])
    return np.asarray(estimator_pub_result(plan, raw).data.evs)


_STATE: dict[str, QuantumCircuit] = {}
_MEAS: dict[str, list[int]] = {}


def _plan(state: QuantumCircuit, measured: list[int], observables, **options):
    qc = _with_estimator_readout(state, measured)
    _STATE[qc.name], _MEAS[qc.name] = state, measured
    pub = EstimatorPub.coerce((qc, observables), precision=PRECISION)
    return EstimatorPlan.from_pub(pub, QMEstimatorOptions(**options))


def _product_state() -> QuantumCircuit:
    qc = QuantumCircuit(4, name="product")
    qc.h(0)  # |+>: <X> = 1
    qc.x(1)  # |1>: <Z> = -1
    qc.ry(0.6, 2)  # <Z> = cos 0.6, <X> = sin 0.6
    qc.rx(0.7, 3)  # <Z> = cos 0.7, <Y> = -sin 0.7
    return qc


OBSERVABLES = ["IIIZ", "IIZI", "IIIX", "IZIX", "IIZX", "YIII", "YZXX", "ZIZI"]


@pytest.mark.parametrize("abelian_grouping", [True, False], ids=["abelian", "no_grouping"])
def test_multi_qubit_expectation_values(abelian_grouping):
    state = _product_state()
    observables = [SparsePauliOp(label) for label in OBSERVABLES] + [
        SparsePauliOp(["IZZX", "IIZX", "IXII"], [0.5, -1.0, 2.0])
    ]
    plan = _plan(state, [0, 1, 2, 3], observables, abelian_grouping=abelian_grouping)
    evs = _estimate(plan)
    exact = np.array([Statevector(state).expectation_value(obs).real for obs in observables])
    np.testing.assert_allclose(evs, exact, atol=TOL)


def test_stream_layout_matches_metadata_without_grouping():
    state = _product_state()
    # Qubit-wise commuting terms: grouped they share one basis, ungrouped they do not.
    plan = _plan(state, [0, 1, 2, 3], [SparsePauliOp(["IIIZ", "IIZI", "IZII"])], abelian_grouping=False)
    assert len(plan.metadata) == 3


def test_parameter_sweep_expectation_values():
    from qiskit.circuit import Parameter

    theta = Parameter("theta")
    state = QuantumCircuit(3, name="sweep")
    state.h(0)
    state.ry(theta, 1)
    state.x(2)
    values = np.array([[[0.0]], [[0.4]], [[1.2]]])  # bindings shape (3, 1) broadcast against 3 observables
    observables = [SparsePauliOp("IZX"), SparsePauliOp("ZXI"), SparsePauliOp("ZZZ")]
    qc = _with_estimator_readout(state, [0, 1, 2])
    pub = EstimatorPub.coerce((qc, observables, values), precision=PRECISION)
    plan = EstimatorPlan.from_pub(pub, QMEstimatorOptions())

    rng_seed = 11
    buffers = [(p, codes) for p, group in enumerate(plan.obs_codes) for codes in group]
    raw = np.zeros((len(buffers), plan.shots, 3), dtype=int)
    rng = np.random.default_rng(rng_seed)
    for b, (p, codes) in enumerate(buffers):
        bound = state.assign_parameters(values.reshape(-1, 1)[p])
        for j, code in enumerate(codes):
            if code == 2:
                bound.sdg(j)
            if code in (1, 2):
                bound.h(j)
        probs = Statevector(bound).probabilities_dict(qargs=[0, 1, 2])
        keys = list(probs)
        for s, d in enumerate(rng.choice(len(keys), size=plan.shots, p=[probs[k] for k in keys])):
            raw[b, s] = [int(c) for c in keys[d][::-1]]
    evs = estimator_pub_result(plan, raw).data.evs

    exact = np.array(
        [
            [Statevector(state.assign_parameters(v)).expectation_value(obs).real for obs in observables]
            for v in values.reshape(-1, 1)
        ]
    )
    np.testing.assert_allclose(evs, exact, atol=TOL)


# ---------------------------------------------------------------------------------------------
# Which qubits the estimator measures
# ---------------------------------------------------------------------------------------------


def _transpiled_with_ancillas():
    """2 virtual qubits (only q0 touched) on a 5-qubit line; returns circuit and layout map."""
    virtual = QuantumCircuit(2)
    virtual.h(0)
    tqc = transpile(virtual, coupling_map=CouplingMap.from_line(5), initial_layout=[3, 1], optimization_level=0)
    return tqc


def test_measured_qubits_include_idle_virtual_skip_unobserved_ancillas():
    from qiskit_qm_provider.backend.backend_utils import estimator_measured_qubits

    tqc = _transpiled_with_ancillas()
    physical_q1 = tqc.layout.final_index_layout(filter_ancillas=True)[1]
    obs = SparsePauliOp("ZX").apply_layout(tqc.layout)  # X on virtual q0, Z on idle virtual q1
    measured = estimator_measured_qubits([obs])
    assert physical_q1 in measured  # idle in the virtual circuit, but observed
    assert measured == sorted(tqc.layout.final_index_layout(filter_ancillas=True))  # ancillas unobserved


def test_measured_qubits_skip_unobserved_active_qubit():
    from qiskit_qm_provider.backend.backend_utils import estimator_measured_qubits

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)  # active but never observed
    measured = estimator_measured_qubits([SparsePauliOp("XIZ")])
    assert measured == [0, 2]


def test_measured_qubits_all_identity_measures_nothing():
    from qiskit_qm_provider.backend.backend_utils import estimator_measured_qubits

    tqc = _transpiled_with_ancillas()
    assert estimator_measured_qubits([SparsePauliOp("II").apply_layout(tqc.layout)]) == []


def test_projector_terms_rejected_with_clear_message():
    from qiskit.quantum_info import SparseObservable
    from qiskit_qm_provider.backend.backend_utils import estimator_measured_qubits

    with pytest.raises(ValueError, match=r"not Pauli strings.*as_paulis"):
        estimator_measured_qubits([SparseObservable("0X")])


def test_identity_terms_evaluated_exactly_without_measurement():
    # <0.5 I + 2 Z_0> and <3 I>: identity terms are never measured (no group, no buffer).
    state = QuantumCircuit(2, name="identity")
    state.x(0)
    observables = [SparsePauliOp(["II", "IZ"], [0.5, 2.0]), SparsePauliOp("II", 3.0)]
    plan = _plan(state, [0], observables)
    assert len(plan.metadata) == 1 and plan.obs_codes == [[(0,)]]
    result = estimator_pub_result(plan, _emulate_stream(plan, state, [0]))
    np.testing.assert_allclose(result.data.evs, [0.5 - 2.0, 3.0], atol=TOL)
    assert result.data.stds[1] == 0.0


def test_identity_only_pub_is_never_run():
    from qiskit.circuit import Parameter

    state = QuantumCircuit(2, name="only_identity")
    state.rx(Parameter("t"), 0)
    pub = EstimatorPub.coerce((_with_estimator_readout(state, []), SparsePauliOp("II", -1.5), [[0.1], [0.2]]), precision=PRECISION)
    plan = EstimatorPlan.from_pub(pub, QMEstimatorOptions(input_type=InputType.INPUT_STREAM))
    assert not plan.has_tasks and not plan.streams_inputs and list(plan._pushes()) == []
    result = estimator_pub_result(plan, None)
    np.testing.assert_allclose(result.data.evs, [-1.5, -1.5])
    np.testing.assert_allclose(result.data.stds, [0.0, 0.0])


def test_idle_observed_qubit_is_measured_not_assumed():
    # q2 is untouched but observed: it is measured (Z -> +1, X -> ~0 from data).
    state = QuantumCircuit(3, name="idle")
    state.h(0)
    state.x(1)
    observables = [SparsePauliOp("ZIX"), SparsePauliOp("XZI"), SparsePauliOp("ZZX")]
    plan = _plan(state, [0, 1, 2], observables)
    evs = _estimate(plan)
    exact = np.array([Statevector(state).expectation_value(obs).real for obs in observables])
    np.testing.assert_allclose(evs, exact, atol=TOL)


def test_measured_qubits_include_observed_ancilla():
    from qiskit_qm_provider.backend.backend_utils import estimator_measured_qubits

    tqc = _transpiled_with_ancillas()
    virtual_q0 = tqc.layout.final_index_layout(filter_ancillas=True)[0]
    ancilla = next(q for q in range(tqc.num_qubits) if q not in tqc.layout.final_index_layout(filter_ancillas=True))
    label = ["I"] * tqc.num_qubits
    label[virtual_q0], label[ancilla] = "X", "Z"
    obs = SparsePauliOp("".join(reversed(label)))  # ISA observable with support on an ancilla
    measured = estimator_measured_qubits([obs])
    assert measured == sorted([virtual_q0, ancilla])


def test_measured_qubits_union_over_observables_array():
    from qiskit_qm_provider.backend.backend_utils import estimator_measured_qubits

    qc = QuantumCircuit(4)
    observables = [[SparsePauliOp("IIIZ")], [SparsePauliOp(["IXII", "IIII"])]]
    assert estimator_measured_qubits(observables) == [0, 2]


def test_ancilla_observable_expectation_value():
    # q2 plays an ancilla: never touched, observed with Z (+1) alongside X on q0 (+1).
    state = QuantumCircuit(3, name="ancilla")
    state.h(0)
    observables = [SparsePauliOp("ZIX"), SparsePauliOp("ZII")]
    plan = _plan(state, [0, 2], observables)
    np.testing.assert_allclose(_estimate(plan), [1.0, 1.0], atol=TOL)
