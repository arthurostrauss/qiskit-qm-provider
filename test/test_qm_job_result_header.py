"""Tests that QMJob Result assembly supports Qiskit Result lookup conventions."""

from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData, MeasLevel

from qiskit_qm_provider.backend.backend_utils import experiment_result_header


def _result_for_circuit(qc: QuantumCircuit, counts: dict[str, int]) -> Result:
    header = experiment_result_header(qc)
    exp = ExperimentResult(
        shots=sum(counts.values()),
        success=True,
        meas_level=MeasLevel.CLASSIFIED,
        data=ExperimentResultData(counts=counts),
        header=header,
    )
    return Result(
        results=[exp],
        backend_name="test_qm",
        backend_version="0.0.0",
        job_id="job-1",
        success=True,
    )


class TestQMJobResultHeaderIntegration:
    def test_get_counts_by_circuit_name(self):
        qc = QuantumCircuit(1, name="named_exp")
        creg = ClassicalRegister(1, "c")
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _result_for_circuit(qc, {"0": 3, "1": 7})

        assert result.get_counts("named_exp") == {"0": 3, "1": 7}

    def test_get_counts_by_circuit_object(self):
        qc = QuantumCircuit(1, name="named_exp")
        creg = ClassicalRegister(1, "c")
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _result_for_circuit(qc, {"0": 1, "1": 1})

        assert result.get_counts(qc) == {"0": 1, "1": 1}

    def test_get_counts_by_index(self):
        qc = QuantumCircuit(1, name="idx_exp")
        creg = ClassicalRegister(1, "c")
        qc.add_register(creg)
        qc.measure(0, creg[0])
        result = _result_for_circuit(qc, {"0": 4})

        assert result.get_counts(0) == {"0": 4}

    def test_counts_formatted_with_creg_sizes(self):
        qc = QuantumCircuit(2, name="split")
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(1, "b")
        qc.add_register(a, b)
        qc.measure(0, a)
        qc.measure(1, b)
        result = _result_for_circuit(qc, {"01": 2, "10": 1})

        assert result.get_counts("split") == {"0 1": 2, "1 0": 1}
