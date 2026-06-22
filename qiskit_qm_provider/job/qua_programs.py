# Copyright 2026 Arthur Strauss
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QUA program builders for sampler and estimator from Qiskit circuits and PUBs.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from ..backend import QMBackend
from ..backend.backend_utils import has_conflicting_calibrations
from qiskit import QuantumCircuit
from qm import Program
from qm.qua import *
from qiskit.primitives.containers.sampler_pub import SamplerPub
from ..parameter_table import ParameterTable, QUA2DArray
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .qm_estimator_job import _ExecutionPlan


def _process_circuit(
    qc: QuantumCircuit,
    backend: QMBackend,
    num_shots: int,
    param_table: Optional[ParameterTable | List[ParameterTable]] = None,
    compute_state_int: bool = True,
    **kwargs,
):
    shot_var = declare(int)

    with for_(shot_var, 0, shot_var < num_shots, shot_var + 1):
        comp = backend.quantum_circuit_to_qua(qc, param_table)
        outputs = comp.outputs
        # Save each register / loose bit to its compiler-owned stream.
        for field in outputs.parameters:
            if compute_state_int:
                save(field.state_int, field.stream)
            elif field.is_array:
                loop_var = declare(int)
                with for_(loop_var, 0, loop_var < field.length, loop_var + 1):
                    save(field.var[loop_var], field.stream)
            else:
                # Scalar output (e.g. a loose clbit) — save the single value directly.
                save(field.var, field.stream)

    return outputs


def _process_sampler_pub(
    pub: SamplerPub,
    backend: QMBackend,
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
        outputs = _process_circuit(
            pub.circuit,
            backend,
            pub.shots,
            **kwargs,
        )
    else:
        param_table.declare()
        p = declare(int)

        with for_(p, 0, p < pub.parameter_values.ravel().size, p + 1):
            if param_table.input_type is None:
                # Declare the parameters at compile time
                param_values_qua = QUA2DArray("param_values", pub.parameter_values.ravel().as_array())
                param_values_qua.declare()
                param_table.assign_parameters(
                    {param.name: param_values_qua[p][i] for i, param in enumerate(param_table.parameters)}
                )
            else:
                param_table.rcv()
            outputs = _process_circuit(
                pub.circuit,
                backend,
                pub.shots,
                param_table,
                **kwargs,
            )
    return outputs


def sampler_program(
    backend: QMBackend,
    pubs: List[SamplerPub],
    param_tables: List[ParameterTable],
    **kwargs,
) -> Program:
    """Return the QUA program for the given PUBs."""
    circuits = [pub.circuit for pub in pubs]
    num_circuits = len(circuits)
    # TODO: Handle OPNIC case where circuits share parameters (loading might not work)
    outputs_tables = []
    with program() as sampler_prog:
        backend.init_macro()

        for i in range(num_circuits):
            outputs = _process_sampler_pub(
                pubs[i],
                backend,
                param_tables[i],
                **kwargs,
            )
            outputs_tables.append(outputs)
        with stream_processing():
            for i, outputs in enumerate(outputs_tables):
                for creg_name, field in outputs.table.items():
                    field.stream.buffer(pubs[i].shots).save_all(f"{creg_name}_{i}")
    return sampler_prog


def _process_observables_with_circuit(
    plan: _ExecutionPlan,
    backend: QMBackend,
    **kwargs,
):
    """
    QUA macro to process observables and execute the circuit.
    Handles both cases where observables_var.input_type is None (compile-time)
    or not None (runtime input).

    Args:
        plan: The ExecutionPlan containing the circuit
        backend: The QMBackend to use
        observables_var: ParameterTable for observables
        param_table: Optional additional ParameterTable to pass to _process_circuit
        **kwargs: Additional arguments for _process_circuit
    """
    obs_length_var = kwargs.get("obs_length_var", None)
    obs_idx = declare(int)
    num_qubits = len(plan.active_qubit_indices)
    plan.observables_var.declare()
    if obs_length_var is not None:
        obs_length_var.declare()
    if plan.observables_var.input_type is None:
        # Flatten all parameter-group observable indices into a single (total_tasks, num_qubits) array
        # so that ONE QUA for loop covers all tasks and all shots flow into the same stream handles.
        all_obs_flat = np.vstack([np.array(g, dtype=np.int32) for g in plan.obs_indices])
        obs_indices_all = QUA2DArray("obs_indices_all", all_obs_flat, qua_type=int)
        obs_indices_all.declare()

        process_param_table = (
            [plan.param_table, plan.observables_var] if plan.param_table is not None else plan.observables_var
        )
        with for_(obs_idx, 0, obs_idx < obs_indices_all.n_rows, obs_idx + 1):
            plan.observables_var.assign_parameters(
                {f"obs_{i}": obs_indices_all[obs_idx, i] for i in range(num_qubits)}
            )
            outputs = _process_circuit(
                plan.pub.circuit,
                backend,
                plan.shots,
                param_table=process_param_table,
                compute_state_int=False,
                **kwargs,
            )
    else:
        if obs_length_var is not None:
            obs_length_var.rcv()
        with for_(obs_idx, 0, obs_idx < obs_length_var.var, obs_idx + 1):
            plan.observables_var.rcv()
            # Combine param_table and observables_var if param_table is provided
            process_param_table = (
                [plan.param_table, plan.observables_var] if plan.param_table is not None else plan.observables_var
            )
            outputs = _process_circuit(
                plan.pub.circuit,
                backend,
                plan.shots,
                param_table=process_param_table,
                compute_state_int=False,
                **kwargs,
            )

    return outputs


def _process_estimator_pub(
    plan: _ExecutionPlan,
    backend: QMBackend,
    **kwargs,
):
    """
    Process the given EstimatorPub with the given backend.
    If the parameter table is not provided, the circuit is processed without parameters.
    If the parameter table is provided, the circuit is processed with parameters.
    This is a QUA macro that is used to process the circuit.
    """
    if plan.param_table is None:
        # Process observables only (no additional param_table)
        outputs = _process_observables_with_circuit(
            plan,
            backend,
            **kwargs,
        )
    else:
        # Process with both param_table and observables_var
        p = declare(int)
        plan.param_table.declare()
        if plan.param_table.input_type is None:
            # Declare the parameters at compile time
            param_values_qua = QUA2DArray(
                f"param_values_{plan.param_table.name}",
                plan.pub.parameter_values.ravel().as_array(),
            )
            param_values_qua.declare()

        with for_(p, 0, p < plan.pub.parameter_values.ravel().size, p + 1):
            if plan.param_table.input_type is None:
                plan.param_table.assign_parameters(
                    {param.name: param_values_qua[p, i] for i, param in enumerate(plan.param_table.parameters)}
                )
            else:
                plan.param_table.rcv()

            # Process observables with the additional param_table
            outputs = _process_observables_with_circuit(
                plan,
                backend,
                **kwargs,
            )
    return outputs


def estimator_program(backend: QMBackend, execution_plans: List[_ExecutionPlan], **kwargs) -> Program:
    """
    Return the QUA program for the estimator primitive based on pre-computed plans.
    """
    # The execution plans are generated in the 'submit' method and passed here.
    outputs_tables = []
    with program() as estimator_prog:
        backend.init_macro()

        # Loop over each PUB/circuit
        for plan in execution_plans:
            outputs = _process_estimator_pub(
                plan,
                backend,
                **kwargs,
            )
            outputs_tables.append(outputs)

        with stream_processing():
            for i, outputs in enumerate(outputs_tables):
                for creg_name, field in outputs.table.items():
                    bool_stream = field.stream.boolean_to_int()
                    # Array register → (shots, n_bits); scalar output (length 0) → (shots,).
                    if field.is_array:
                        bool_stream.buffer(execution_plans[i].shots, field.length).save_all(f"{creg_name}_{i}")
                    else:
                        bool_stream.buffer(execution_plans[i].shots).save_all(f"{creg_name}_{i}")

    return estimator_prog


def get_run_program(backend: QMBackend, num_shots, circuits: List[QuantumCircuit]) -> Program | List[Program]:
    """
    QUA program generated upon the call of backend.run().
    If the circuits have conflicting calibrations, a list of programs is returned.
    Otherwise, a single program is returned, executing all the circuits sequentially.
    Args:
        backend: The QMBackend to use
        num_shots: Number of shots to execute
        circuits: List of QuantumCircuits to execute

    Returns:
        Program: The QUA program
    """
    outputs_tables = []
    if not has_conflicting_calibrations(circuits):
        with program() as prog:
            backend.init_macro()

            for i, qc in enumerate(circuits):
                outputs = _process_circuit(
                    qc,
                    backend,
                    num_shots,
                )
                outputs_tables.append(outputs)

            with stream_processing():
                for i, outputs in enumerate(outputs_tables):
                    for creg_name, field in outputs.table.items():
                        field.stream.save_all(f"{creg_name}_{i}")

        return prog
    else:
        progs = []
        for j, qc in enumerate(circuits):
            with program() as prog:
                backend.init_macro()

                outputs = _process_circuit(
                    qc,
                    backend,
                    num_shots,
                )
                outputs_tables.append(outputs)

                with stream_processing():
                    for creg_name, field in outputs_tables[-1].table.items():
                        field.stream.save_all(f"{creg_name}_{j}")
            progs.append(prog)

        return progs
