from __future__ import annotations

from collections import defaultdict

import numpy as np
from qiskit.circuit import Parameter
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from ..backend import QMBackend
from ..backend.backend_utils import _QASM3_DUMP_LOOSE_BIT_PREFIX, has_conflicting_calibrations
from qiskit import QuantumCircuit
from quam.utils.qua_types import QuaScalarInt
from qm import Program
from qm.qua import *
from qm.qua._dsl import _ResultSource
from qiskit.primitives.containers.sampler_pub import SamplerPub
from ..parameter_table import InputType, ParameterTable
from typing import List, Optional


def _process_circuit(
    qc: QuantumCircuit,
    backend: QMBackend,
    num_shots: int,
    state_int: QuaScalarInt,
    shot_var: QuaScalarInt,
    reg_streams: List[_ResultSource],
    solo_bits_stream: Optional[_ResultSource] = None,
    param_table: Optional[ParameterTable | List[ParameterTable]] = None,
    **kwargs,
):
    with for_(shot_var, 0, shot_var < num_shots, shot_var + 1):
        result = backend.quantum_circuit_to_qua(qc, param_table)

        clbits_dict = {creg.name: [result.result_program[creg.name][i] for i in range(creg.size)] for creg in qc.cregs}
        num_solo_bits = len([bit for bit in qc.clbits if len(qc.find_bit(bit).registers) == 0])
        if num_solo_bits > 0:
            if solo_bits_stream is None:
                raise ValueError("Circuit contains bits without registers but no stream provided")
            clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX] = [
                result.result_program[f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{i}"] for i in range(num_solo_bits)
            ]
        # Save integer state to each stream

        for creg, stream in zip(qc.cregs, reg_streams):
            assign(
                state_int,
                sum(
                    (state_int + (1 << i) * Cast.to_int(clbits_dict[creg.name][i]) for i in range(1, creg.size)),
                    start=Cast.to_int(clbits_dict[creg.name][0]),
                ),
            )
            save(state_int, stream)
        if num_solo_bits > 0:
            assign(
                state_int,
                sum(
                    (
                        state_int + (1 << i) * Cast.to_int(clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX][i])
                        for i in range(1, num_solo_bits)
                    ),
                    start=Cast.to_int(clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX][0]),
                ),
            )
            save(state_int, solo_bits_stream)


def _process_sampler_pub(
    pub: SamplerPub,
    backend: QMBackend,
    state_int: QuaScalarInt,
    shot: QuaScalarInt,
    regs_streams: List[_ResultSource],
    solo_bits_stream: Optional[_ResultSource] = None,
    param_table: Optional[ParameterTable] = None,
    **kwargs,
):
    """
    Process the given PUB with the given backend.
    If the parameter table is not provided, the circuit is processed without parameters.
    If the parameter table is provided, the circuit is processed with parameters.
    This is a QUA macro that is used to process the circuit.
    """
    if param_table is None:
        _process_circuit(
            pub.circuit,
            backend,
            pub.shots,
            state_int,
            shot,
            regs_streams,
            solo_bits_stream,
            **kwargs,
        )
    else:
        p = declare(int)
        with for_(p, 0, p < pub.parameter_values.ravel().size, p + 1):
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
                **kwargs,
            )


def sampler_program(
    backend: QMBackend, pubs: List[SamplerPub], input_type: Optional[InputType] = None, **kwargs
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
            len([bit for bit in circuits[0].clbits if len(circuits[i].find_bit(bit).registers) == 0])
            for i in range(num_circuits)
        ]
        regs_streams = [[declare_stream() for _ in range(num_cregs)] for num_cregs in num_registers]
        solo_bits_stream = [declare_stream() for _ in range(num_circuits)]

        if backend.init_macro:
            backend.init_macro()

        for i in range(num_circuits):
            if param_tables[i] is not None:
                param_tables[i].declare_variables()
            _process_sampler_pub(
                pubs[i],
                backend,
                state_int,
                shot,
                regs_streams[i],
                solo_bits_stream[i] if num_solo_bits[i] > 0 else None,
                param_tables[i],
                **kwargs,
            )

        with stream_processing():
            for i, creg_streams in enumerate(regs_streams):
                for creg, creg_stream in zip(circuits[i].cregs, creg_streams):
                    creg_stream.save_all(f"{creg.name}_{i}")

    return sampler_prog


def estimator_program(backend: QMBackend, pubs: List[EstimatorPub], input_type: InputType, **kwargs) -> Program:
    """
    Return the QUA program for the estimator primitive based on pre-computed plans.
    """
    from .qm_estimator_job import _ExecutionPlan

    # The execution plans are generated in the 'submit' method and passed here.
    execution_plans: List[_ExecutionPlan] = kwargs["execution_plans"]

    circuits = [pub.circuit for pub in pubs]

    # Define the ParameterTables once per circuit.
    param_tables = [
        ParameterTable.from_qiskit(qc, input_type=input_type, filter_function=lambda p: isinstance(p, Parameter))
        for qc in circuits
    ]
    observables_vars = [
        ParameterTable.from_qiskit(
            qc,
            input_type=input_type,
            filter_function=lambda p: "obs" in p.name and qc.has_var(p.name),
        )
        for qc in circuits
    ]

    results_streams = [declare_stream() for _ in pubs]

    with program() as estimator_prog:
        shot = declare(int)
        state_int = declare(int, value=0)

        # This is the main loop counter for the flattened tasks in the plan.
        task_idx = declare(int)

        if backend.init_macro:
            backend.init_macro()

        # Loop over each PUB/circuit
        for i, pub in enumerate(pubs):
            plan = execution_plans[i]

            # This is the single, powerful QUA loop that will run through all tasks.
            # The 'submit' method will push the correct data for each 'task_idx'.
            with for_(task_idx, 0, task_idx < plan.total_tasks, task_idx + 1):
                # These methods now implicitly wait for the 'push_to_opx' call
                # in the submit method for the current 'task_idx'.
                if param_tables[i] is not None:
                    param_tables[i].load_input_values()
                observables_vars[i].load_input_values()

                # The _process_circuit function remains the same.
                _process_circuit(
                    circuits[i],
                    backend,
                    plan.shots_per_task,
                    state_int,
                    shot,
                    [results_streams[i]],
                    param_table=[param_tables[i], observables_vars[i]],
                )

        with stream_processing():
            for i, stream in enumerate(results_streams):
                # The name now reflects that it contains all counts for that pub
                stream.save_all(f"counts_pub_{i}")

    return estimator_prog


def get_run_program(backend: QMBackend, num_shots, circuits: List[QuantumCircuit]) -> Program | List[Program]:
    num_circuits = len(circuits)

    def _process_circuit(
        qc: QuantumCircuit,
        state_int: QuaScalarInt,
        shot_var: QuaScalarInt,
        reg_streams: List[_ResultSource],
        solo_bits_stream: Optional[_ResultSource] = None,
    ):
        with for_(shot_var, 0, shot_var < num_shots, shot_var + 1):
            result = backend.qiskit_to_qua_macro(qc)

            clbits_dict = {
                creg.name: [result.result_program[creg.name][i] for i in range(creg.size)] for creg in qc.cregs
            }
            num_solo_bits = len([bit for bit in qc.clbits if len(qc.find_bit(bit).registers) == 0])
            if num_solo_bits > 0:
                if solo_bits_stream is None:
                    raise ValueError("Circuit contains bits without registers but no stream provided")
                clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX] = [
                    result.result_program[f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{i}"] for i in range(num_solo_bits)
                ]
            # Save integer state to each stream

            for creg, stream in zip(qc.cregs, reg_streams):
                assign(
                    state_int,
                    sum(
                        (state_int + (1 << i) * Cast.to_int(clbits_dict[creg.name][i]) for i in range(1, creg.size)),
                        start=Cast.to_int(clbits_dict[creg.name][0]),
                    ),
                )
                save(state_int, stream)
            if num_solo_bits > 0:
                assign(
                    state_int,
                    sum(
                        (
                            state_int + (1 << i) * Cast.to_int(clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX][i])
                            for i in range(1, num_solo_bits)
                        ),
                        start=Cast.to_int(clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX][0]),
                    ),
                )
                save(state_int, solo_bits_stream)

    if not has_conflicting_calibrations(circuits):
        with program() as prog:
            if backend.init_macro:
                backend.init_macro()

            shot = declare(int)
            state_int = declare(int, value=0)
            num_registers = [len(circuits[i].cregs) for i in range(num_circuits)]

            num_solo_bits = [
                len([bit for bit in circuits[0].clbits if len(circuits[i].find_bit(bit).registers) == 0])
                for i in range(num_circuits)
            ]
            regs_streams = [[declare_stream() for _ in range(num_cregs)] for num_cregs in num_registers]
            solo_bits_stream = [declare_stream() for _ in range(num_circuits)]

            for i, qc in enumerate(circuits):
                _process_circuit(
                    qc,
                    state_int,
                    shot,
                    regs_streams[i],
                    solo_bits_stream[i] if num_solo_bits[i] > 0 else None,
                )

            with stream_processing():
                for i, creg_streams in enumerate(regs_streams):
                    for creg, creg_stream in zip(circuits[i].cregs, creg_streams):
                        creg_stream.save_all(f"{creg.name}_{i}")

        return prog
    else:
        progs = []
        for j, qc in enumerate(circuits):
            with program() as prog:
                if backend.init_macro:
                    backend.init_macro()
                shot = declare(int)
                state_int = declare(int, value=0)
                num_registers = len(qc.cregs)
                num_solo_bits = len([bit for bit in qc.clbits if len(qc.find_bit(bit).registers) == 0])
                regs_streams = [declare_stream() for _ in range(num_registers)]
                solo_bits_stream = declare_stream() if num_solo_bits > 0 else None
                _process_circuit(
                    qc,
                    state_int,
                    shot,
                    regs_streams,
                    solo_bits_stream,
                )

            with stream_processing():
                for i, creg_stream in enumerate(regs_streams):
                    creg_stream.save_all(f"{qc.cregs[i].name}_{j}")

            progs.append(prog)
        return progs
