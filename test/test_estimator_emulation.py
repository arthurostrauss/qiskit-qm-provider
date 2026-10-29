"""End-to-end check of QMEstimatorV2 against Qiskit's StatevectorEstimator, emulating only the OPX.

Unlike ``test_estimator_correctness``, nothing about the readout is assumed. The PUB goes through
the real ``QMEstimatorV2.validate_estimator_pubs`` (on a real backend) and
``EstimatorPlan.from_pub``. Then, per binding and per group, as the QUA program does:

* the observable row is the packed row the program receives (only the used prefix over IO), and
  the basis codes are unpacked from it with the same shifts and masks;
* the validated circuit is interpreted as it stands: each ``__obs_j`` switch runs the case body
  its code selects, and bit ``j`` of the stream is the qubit that circuit measures into ``__c[j]``.

The outcomes feed the real ``estimator_pub_result``. ``test_generated_qua_follows_readout`` checks
the remaining link: the QUA the backend generates unpacks codes and fills ``__c`` in that order.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pytest

pytest.importorskip("quam_builder.builder.superconducting.modify_quam")

from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit import CASE_DEFAULT, ControlFlowOp, Parameter, ParameterVector, SwitchCaseOp  # noqa: E402
from qiskit.circuit.classical import expr  # noqa: E402
from qiskit.primitives import StatevectorEstimator  # noqa: E402
from qiskit.primitives.containers.estimator_pub import EstimatorPub  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

from qiskit_qm_provider.job.primitive_plan import EstimatorPlan
from qiskit_qm_provider.job.primitive_plan.estimator import (  # noqa: E402
    _estimator_readout,
    estimator_basis_input,
)
from qiskit_qm_provider.job.pub_parameters import OBS_QUBITS_PER_WORD, obs_row_payload  # noqa: E402
from qiskit_qm_provider.job.qm_estimator_job import estimator_pub_result  # noqa: E402
from qiskit_qm_provider.parameter_table import InputType, ParameterPool  # noqa: E402
from qiskit_qm_provider.primitives.qm_estimator import QMEstimatorOptions, QMEstimatorV2  # noqa: E402

PRECISION = 0.01  # 10_000 shots
_SKIPPED = {"reset", "barrier", "delay"}


@pytest.fixture(scope="module")
def backend_factory(tmp_path_factory):
    from ._in_memory_quam import build_backend

    backends = {}

    def make(num_qubits):
        if num_qubits not in backends:
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("QUAM_STATE_PATH", str(tmp_path_factory.mktemp("quam_state")))
                backends[num_qubits] = build_backend(num_qubits, conditional_plays=())
        return backends[num_qubits]

    return make


def _interpret(circuit, out, qubit_map, codes):
    """Append ``circuit``'s gates to ``out``, resolving each ``__obs_j`` switch with ``codes[j]``.

    Only the ``__c`` readout measurements are allowed (the outcomes are read from the final
    state): the emulator does not model a measurement collapsing the state mid-circuit.
    """
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [qubit_map[q] for q in instruction.qubits]
        if operation.name == "measure":
            assert circuit.find_bit(instruction.clbits[0]).registers[0][0].name == "__c", "measurement outside __c"
            continue
        if operation.name in _SKIPPED:
            continue
        if isinstance(operation, SwitchCaseOp):
            target = operation.target
            assert isinstance(target, expr.Var) and target.name.startswith("__obs_"), target
            value = codes[int(target.name[len("__obs_"):])]
            cases = list(operation.cases_specifier())
            block = next((b for values, b in cases if value in values), None)
            if block is None:
                block = next(b for values, b in cases if CASE_DEFAULT in values)
            _interpret(block, out, dict(zip(block.qubits, qubits)), codes)
            continue
        assert not isinstance(operation, ControlFlowOp), f"unexpected control flow {operation.name}"
        out.append(operation, qubits)


def _unpack(row, base, num_qubits):
    """The program's ``assign_obs_codes``: 2 bits per qubit, ``OBS_QUBITS_PER_WORD`` per word."""
    return [
        (row[base + q // OBS_QUBITS_PER_WORD] >> (2 * (q % OBS_QUBITS_PER_WORD))) & 3
        for q in range(num_qubits)
    ]


def _samples(probabilities, shots, rng):
    """Outcomes drawn from ``probabilities`` (index bit ``k`` = measured qubit ``k``): exact
    frequencies (to 1/shots) on a small support, sampled otherwise (rounding is biased when most
    outcomes are rarer than 1/shots)."""
    num_bits = int(np.log2(len(probabilities)))
    support = np.flatnonzero(probabilities > 1e-12)
    p = probabilities[support]
    if len(support) <= shots // 50:
        counts = np.floor(p * shots).astype(int)
        counts[np.argsort(counts - p * shots)[: shots - counts.sum()]] += 1
    else:
        counts = rng.multinomial(shots, p / p.sum())
    bits = (support[:, None] >> np.arange(num_bits)) & 1
    return np.repeat(bits, counts, axis=0)


def _emulate_stream(plan, rng):
    """The ``__c`` stream the plan's QUA program produces, buffer by buffer."""
    if not plan.has_tasks:
        return None
    circuit = plan.pub.circuit
    measured = _estimator_readout(circuit)
    words = plan.words_per_obs
    buffers = []
    for p in range(plan.num_bindings):
        packed = plan.obs_rows[0 if plan.shared_obs_row else p]
        if plan.streams_obs:  # what reaches the controller (IO: the used prefix only)
            payload = obs_row_payload(plan.obs_row, packed, words)
            row = payload + [0] * (len(packed.wire) - len(payload))
        else:
            row = list(packed.wire)
        bound = circuit.assign_parameters(dict(zip(circuit.parameters, plan.binding_rows[p])))
        for group in range(row[0]):
            state = QuantumCircuit(circuit.num_qubits)
            _interpret(bound, state, dict(zip(bound.qubits, state.qubits)), _unpack(row, 1 + group * words, len(measured)))
            buffers.append(_samples(Statevector(state).probabilities(qargs=measured), plan.shots, rng))
    return np.stack(buffers)


def _estimate(backend, pub_like, input_type, abelian_grouping, seed=0):
    ParameterPool.reset()
    options = QMEstimatorOptions(input_type=input_type, abelian_grouping=abelian_grouping)
    estimator = QMEstimatorV2(backend, options)
    (pub,) = estimator.validate_estimator_pubs([EstimatorPub.coerce(pub_like, PRECISION)])
    plan = EstimatorPlan.from_pub(pub, options, plan_index=0)
    result = estimator_pub_result(plan, _emulate_stream(plan, np.random.default_rng(seed)))
    return result, plan


def _random_case(backend, rng, num_qubits, num_params, param_shape, obs_shape, vector):
    circuit = QuantumCircuit(num_qubits)
    params = ParameterVector("t", num_params) if vector else [Parameter(f"p{i}") for i in range(num_params)]
    k = 0
    for layer in range(3):
        for q in range(num_qubits):
            circuit.ry(params[k % num_params] if num_params else rng.uniform(0, 3), q)
            circuit.rz(rng.uniform(-3, 3), q)
            k += 1
        for q in range(layer % 2, num_qubits - 1, 2):
            circuit.cx(q, q + 1)
    isa = transpile(circuit, backend, optimization_level=1, seed_transpiler=int(rng.integers(1_000_000)))

    def observable():
        terms = [("".join(rng.choice(list("IXYZ"), num_qubits)), rng.uniform(-1, 1)) for _ in range(3)]
        return SparsePauliOp.from_list([*terms, ("I" * num_qubits, 0.25)]).apply_layout(isa.layout)

    if obs_shape == ():
        observables = observable()
    else:
        observables = np.empty(obs_shape, dtype=object)
        for index in np.ndindex(obs_shape):
            observables[index] = observable()
    if not num_params:
        return isa, (isa, observables)
    return isa, (isa, observables, rng.uniform(-3, 3, size=(*param_shape, num_params)))


CASES = {
    "shared_row": (2, 2, (3,), ()),  # one observable for every binding: one row
    "per_binding_rows": (3, 3, (3,), (3,)),  # distinct observables per binding
    "broadcast": (4, 2, (2, 1), (3,)),  # (2, 3): each binding measures the union of 3 observables
    "no_parameters": (3, 0, (), (2, 2)),
}


@pytest.mark.parametrize("vector", [True, False], ids=["vector", "scalars"])
@pytest.mark.parametrize("case", list(CASES))
def test_matches_statevector_estimator(backend_factory, case, vector):
    num_qubits, num_params, param_shape, obs_shape = CASES[case]
    if not num_params and not vector:
        pytest.skip("no parameters")
    backend = backend_factory(4)
    rng = np.random.default_rng(sum(map(ord, case)) + vector)
    isa, pub_like = _random_case(backend, rng, num_qubits, num_params, param_shape, obs_shape, vector)
    reference = np.asarray(StatevectorEstimator().run([pub_like]).result()[0].data.evs)
    for input_type, grouping in itertools.product([None, InputType.INPUT_STREAM, InputType.IO1], [True, False]):
        result, plan = _estimate(backend, pub_like, input_type, grouping)
        evs = np.asarray(result.data.evs)
        assert evs.shape == reference.shape
        # Near-exact counts: the residual is the 1/shots resolution of the outcome frequencies.
        np.testing.assert_allclose(evs, reference, atol=1e-3, err_msg=f"{input_type}, grouping={grouping}")


@pytest.mark.parametrize("input_type", [None, InputType.IO1], ids=["compile_time", "io"])
def test_rows_spanning_two_words(backend_factory, input_type):
    # 17 measured qubits: each observable takes 2 words (15 qubits per word).
    num_qubits = 17
    backend = backend_factory(num_qubits)
    rng = np.random.default_rng(5)
    theta = ParameterVector("t", 2)
    circuit = QuantumCircuit(num_qubits)
    circuit.ry(theta[0], 0)
    for q in range(num_qubits - 1):
        circuit.cx(q, q + 1)
    circuit.rx(theta[1], num_qubits - 1)
    isa = transpile(circuit, backend, optimization_level=0)
    labels = ["Z" * num_qubits, "X" * num_qubits, "".join(rng.choice(list("XYZ"), num_qubits)), "Y" + "I" * 15 + "Z"]
    observables = SparsePauliOp.from_list([(label, rng.uniform(-1, 1)) for label in labels]).apply_layout(isa.layout)
    values = rng.uniform(-3, 3, size=(2, 2))
    reference = StatevectorEstimator().run([(isa, observables, values)]).result()[0].data.evs
    result, plan = _estimate(backend, (isa, observables, values), input_type, True)
    assert plan.num_qubits == num_qubits and plan.words_per_obs == 2
    # Wide distributions are sampled: compare within the estimator's own standard errors.
    assert np.all(np.abs(result.data.evs - reference) <= 4 * result.data.stds)


def test_generated_qua_follows_readout(backend_factory):
    from qm import generate_qua_script

    from qiskit_qm_provider.job.qm_estimator_job import QMEstimatorJob

    backend = backend_factory(3)
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    isa = transpile(circuit, backend, optimization_level=0)
    ParameterPool.reset()
    estimator = QMEstimatorV2(backend)
    # Y on q0 and X on q2 (little-endian label); q1 is not measured.
    (pub,) = estimator.validate_estimator_pubs([EstimatorPub.coerce((isa, SparsePauliOp("XIY")), 0.1)])
    job = QMEstimatorJob(backend, [pub], None, switch_obs_circuit=estimator._switch_obs_circuit, abelian_grouping=True)
    plan = job.runtime_pubs[0]
    assert _estimator_readout(plan.pub.circuit) == [0, 2]
    assert [list(row.wire) for row in plan.obs_rows] == [[1, 2 | (1 << 2)]]  # one group: __obs_0 = 2 (Y), __obs_1 = 1 (X)
    assert [estimator_basis_input(j) for j in range(2)] == ["__obs_0", "__obs_1"]

    script = generate_qua_script(job.programs[0])
    unpacked = re.findall(r"assign\((v\d+), \(?\(?(v\d+)(?:>>(\d+))?\)?&3\)", script)
    assert [shift or "0" for _, _, shift in unpacked] == ["0", "2"]  # __obs_0 then __obs_1
    # Each qubit's measurement fills the __c cell of its position in the readout order.
    measured = re.findall(r"measure\('readout', 'q(\d)\.resonator'.*?assign\((a\d+)\[(\d)\]", script, flags=re.S)
    assert [(qubit, cell) for qubit, _, cell in measured] == [("0", "0"), ("2", "1")]
    assert len({array for _, array, _ in measured}) == 1
    assert re.search(r"\.buffer\(100, 2\)\.save_all\(\"__c_0\"\)", script)


@pytest.mark.parametrize("kind", ["register", "input", "var"])
def test_reserved_prefix_rejected(backend_factory, kind):
    from qiskit.circuit import ClassicalRegister
    from qiskit.circuit.classical import types

    backend = backend_factory(2)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    isa = transpile(circuit, backend, optimization_level=0)
    if kind == "register":
        isa.add_register(ClassicalRegister(1, "__mine"))
    elif kind == "input":
        isa.add_input("__obs_0", types.Uint(4))
    else:
        isa.add_var("__flag", True)
    estimator = QMEstimatorV2(backend)
    with pytest.raises(ValueError, match="start with '__' are reserved"):
        estimator.validate_estimator_pubs([EstimatorPub.coerce((isa, SparsePauliOp("ZZ")), 0.1)])


def test_circuit_kept_and_readout_appended(backend_factory):
    # As in BackendEstimatorV2: the PUB circuit is kept as given (its own measurements and
    # registers included) and the __c readout is appended after it, without any warning.
    import warnings

    backend = backend_factory(2)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    isa = transpile(circuit, backend, optimization_level=0)
    estimator = QMEstimatorV2(backend)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (pub,) = estimator.validate_estimator_pubs([EstimatorPub.coerce((isa, SparsePauliOp("XX")), 0.1)])
    prepared = pub.circuit
    assert prepared.layout == isa.layout
    assert [creg.name for creg in prepared.cregs] == ["meas", "__c"]
    readout = prepared.cregs[1]
    measures = [i for i in prepared.data if i.operation.name == "measure"]
    user, appended = measures[:2], measures[2:]
    assert all(clbit in prepared.cregs[0] for i in user for clbit in i.clbits)
    assert [prepared.find_bit(i.qubits[0]).index for i in appended] == [0, 1]
    assert all(clbit in readout for i in appended for clbit in i.clbits)
    assert prepared.data.index(user[-1]) < prepared.data.index(appended[0])


@pytest.mark.parametrize("job_kind", ["local", "iqcc"])
def test_identity_only_job_sends_nothing_and_returns_coefficients(backend_factory, monkeypatch, job_kind):
    from qiskit_qm_provider.job.qm_estimator_job import IQCCEstimatorJob, QMEstimatorJob

    backend = backend_factory(2)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    isa = transpile(circuit, backend, optimization_level=0)
    ParameterPool.reset()
    estimator = QMEstimatorV2(backend)
    pubs = estimator.validate_estimator_pubs([EstimatorPub.coerce((isa, SparsePauliOp("II", 0.5)), 0.1)])

    def no_qm(_):
        raise AssertionError("an identity-only job must not reach the QM")

    monkeypatch.setattr(type(backend), "qm", property(no_qm))
    job_class = QMEstimatorJob if job_kind == "local" else IQCCEstimatorJob
    job = job_class(backend, pubs, None, switch_obs_circuit=estimator._switch_obs_circuit, abelian_grouping=True)
    assert job.programs == []
    job.submit()
    assert float(job.result()[0].data.evs) == 0.5

