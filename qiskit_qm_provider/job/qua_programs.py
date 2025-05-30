from ..backend import QMBackend
from ..backend.backend_utils import _QASM3_DUMP_LOOSE_BIT_PREFIX, validate_circuits
from qiskit import QuantumCircuit
from quam.utils.qua_types import QuaScalar
from qm import Program
from qm.qua import *
from qm.qua._dsl import _ResultSource
from qiskit.primitives.containers.sampler_pub import SamplerPub
from ..parameter_table import InputType, ParameterTable
from typing import List, Optional, Any


def _process_circuit(
    qc: QuantumCircuit,
    backend: QMBackend,
    num_shots: int,
    state_int: QuaScalar[int],
    shot_var: QuaScalar[int],
    reg_streams: List[_ResultSource],
    solo_bits_stream: Optional[_ResultSource] = None,
    param_table: Optional[ParameterTable] = None,
):
    with for_(shot_var, 0, shot_var < num_shots, shot_var + 1):
        result = backend.quantum_circuit_to_qua(qc, param_table)

        clbits_dict = {
            creg.name: [result.result_program[creg.name][i] for i in range(creg.size)]
            for creg in qc.cregs
        }
        num_solo_bits = len([bit for bit in qc.clbits if len(qc.find_bit(bit).registers) == 0])
        if num_solo_bits > 0:
            if solo_bits_stream is None:
                raise ValueError("Circuit contains bits without registers but no stream provided")
            clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX] = [
                result.result_program[f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{i}"]
                for i in range(num_solo_bits)
            ]
        # Save integer state to each stream

        for creg, stream in zip(qc.cregs, reg_streams):
            for i in range(creg.size):
                assign(state_int, state_int + (1 << i) * Cast.to_int(clbits_dict[creg.name][i]))
            save(state_int, stream)
            assign(state_int, 0)
        if num_solo_bits > 0:
            for i in range(num_solo_bits):
                assign(
                    state_int,
                    state_int
                    + (1 << i) * Cast.to_int(clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX][i]),
                )
            save(state_int, solo_bits_stream)
            assign(state_int, 0)


def _process_pub(
    pub: SamplerPub,
    backend: QMBackend,
    state_int: QuaScalar[int],
    shot: QuaScalar[int],
    regs_streams: List[_ResultSource],
    solo_bits_stream: Optional[_ResultSource] = None,
    param_table: Optional[ParameterTable] = None,
):
    """
    Process the given PUB with the given backend.
    If the parameter table is not provided, the circuit is processed without parameters.
    If the parameter table is provided, the circuit is processed with parameters.
    This is a QUA macro that is used to process the circuit.
    """
    if param_table is None:
        _process_circuit(
            pub.circuit, backend, pub.shots, state_int, shot, regs_streams, solo_bits_stream
        )
    else:
        p = declare(int)
        num_param_values = declare(int, value=pub.parameter_values.ravel().as_array().shape[0])
        with for_(p, 0, p < num_param_values, p + 1):
            param_table.load_input_values()
            _process_circuit(
                pub.circuit,
                backend,
                pub.shots,
                state_int,
                shot,
                regs_streams,
                solo_bits_stream,
                param_table,
            )


def sampler_program(
    backend: QMBackend, pubs: List[SamplerPub], input_type: Optional[InputType] = None
) -> Program:
    """Return the QUA program for the given PUBs."""
    circuits = [pub.circuit for pub in pubs]
    num_circuits = len(circuits)
    # TODO: Handle DGX Quantum case where circuits share parameters (loading might not work)
    param_tables = [ParameterTable.from_qiskit(qc, input_type=input_type) for qc in circuits]

    with program() as sampler_prog:
        shot = declare(int)
        state_int = declare(int, value=0)
        num_registers = [len(circuits[i].cregs) for i in range(num_circuits)]
        num_solo_bits = [
            len(
                [bit for bit in circuits[0].clbits if len(circuits[i].find_bit(bit).registers) == 0]
            )
            for i in range(num_circuits)
        ]
        regs_streams = [[declare_stream() for _ in range(num_cregs)] for num_cregs in num_registers]
        solo_bits_stream = [declare_stream() for _ in range(num_circuits)]

        for param_table in param_tables:
            if param_table is not None:
                param_table.declare_variables(declare_streams=False)

        if backend.init_macro:
            backend.init_macro()
        if num_circuits == 1:
            _process_pub(
                pubs[0],
                backend,
                state_int,
                shot,
                regs_streams[0],
                solo_bits_stream[0] if num_solo_bits[0] > 0 else None,
                param_tables[0],
            )
        else:
            qc_var = declare(int)
            with for_(qc_var, 0, qc_var < num_circuits, qc_var + 1):
                with switch_(qc_var):
                    for i in range(num_circuits):
                        with case_(i):
                            _process_pub(
                                pubs[i],
                                backend,
                                state_int,
                                shot,
                                regs_streams[i],
                                solo_bits_stream[i] if num_solo_bits[i] > 0 else None,
                                param_tables[i],
                            )

        with stream_processing():
            for i, creg_streams in enumerate(regs_streams):
                for creg, creg_stream in zip(circuits[i].cregs, creg_streams):
                    creg_stream.save_all(f"{creg.name}_{i}")

    return sampler_prog
