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

"""Backend utilities: circuit-to-QUA translation, calibration handling, measurement outcomes.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

import warnings
from typing import Any, List, TYPE_CHECKING, Dict, Type

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.controlflow import (
    ControlFlowOp,
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    SwitchCaseOp,
)
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Pauli, PauliList

from quam.components import Qubit, QubitPair
from quam.core import QuamRoot
from quam.utils.qua_types import QuaVariableInt
from qm import generate_qua_script
from qm.qua import assign, Cast
from ..additional_gates import CRGate, FSimGate, SYGate, SYdgGate
# Re-exported for backward compatibility; canonical home is quam_macros.superconducting.
from ..quam_macros.superconducting import add_basic_macros

if TYPE_CHECKING:
    from qm_qasm import CompilationResult
    from ..parameter_table import ParameterTable
    from .qm_backend import QMBackend
try:
    from qiskit.circuit.controlflow import get_control_flow_name_mapping

    control_flow_name_mapping = get_control_flow_name_mapping()
except ImportError:
    warnings.warn(
        "get_control_flow_name_mapping is not available in this version of Qiskit, skipping it from control flow mapping."
    )
    control_flow_name_mapping: Dict[str, Type[ControlFlowOp]] = {
        "if_else": IfElseOp,
        "while_loop": WhileLoopOp,
        "for_loop": ForLoopOp,
        "switch_case": SwitchCaseOp,
    }

qasm3_keyword_instructions = (
    "measure",
    "reset",
    "delay",
    "nop",
    "box",
    "for_loop",
    "while_loop",
    "if_else",
    "switch_case",
)
_QASM3_DUMP_LOOSE_BIT_PREFIX = "_bit"


def validate_machine(machine) -> QuamRoot:
    """Validate a QuAM instance before use with the backend.

    Args:
        machine: QuAM instance to validate. Must expose ``qubits`` and
            ``qubit_pairs`` containing :class:`~quam.components.Qubit` and
            :class:`~quam.components.QubitPair` objects respectively.

    Returns:
        The validated QuAM instance.

    Raises:
        ValueError: If required attributes are missing or have wrong types.
    """
    if not hasattr(machine, "qubits") or not hasattr(machine, "qubit_pairs"):
        raise ValueError("Invalid QuAM instance provided, should have qubits and qubit_pairs attributes")
    if not all(isinstance(qubit, Qubit) for qubit in machine.qubits.values()):
        raise ValueError("All qubits should be of type Qubit")
    if not all(isinstance(qubit_pair, QubitPair) for qubit_pair in machine.qubit_pairs.values()):
        raise ValueError("All qubit pairs should be of type QubitPair")

    return machine


def validate_circuits(
    circuits: QuantumCircuit | List[QuantumCircuit],
    should_reset: bool = True,
    check_for_params: bool = False,
) -> List[QuantumCircuit]:
    """Validate circuits before compilation.

    Args:
        circuits: Single circuit or list of circuits to validate.
        should_reset: When ``True``, prepend a reset to circuits that lack one
            at the boundary.
        check_for_params: When ``True``, reject circuits with compile-time
            parameters.

    Returns:
        Validated circuits, with an automatic reset prepended when requested.

    Raises:
        ValueError: If inputs are invalid or classical-bit layout is unsupported.
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    if not all(isinstance(qc, QuantumCircuit) for qc in circuits):
        raise ValueError("Input should be a list of QuantumCircuits")
    if check_for_params and not all(len(qc.parameters) == 0 for qc in circuits):
        raise ValueError("Input should not contain parameters")

    new_circuits = []
    for qc in circuits:
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) > 1:
                raise ValueError("Only one register per clbit is supported.")
        if not has_reset_at_boundary(qc) and should_reset:
            qc_reset = qc.copy_empty_like(vars_mode="drop")
            active_qubits = logically_active_qubits(qc)
            qc_reset.reset(active_qubits)
            new_circuits.append(qc.compose(qc_reset, inplace=False))
        else:
            new_circuits.append(qc)

    return new_circuits


def require_classified_meas_level(meas_level, *, context: str = "") -> None:
    """Raise if ``meas_level`` is not classified 0/1 readout.

    Only :data:`~qiskit.result.models.MeasLevel.CLASSIFIED` is supported end-to-end
    for sampler and ``backend.run()`` result assembly.
    """
    from qiskit.result.models import MeasLevel

    if meas_level != MeasLevel.CLASSIFIED:
        suffix = f" ({context})" if context else ""
        raise NotImplementedError(
            f"Only MeasLevel.CLASSIFIED measurement is supported{suffix}; got {meas_level!r}."
        )


def measurement_output_bit_sizes(qc: QuantumCircuit) -> dict[str, int]:
    """Map measurement stream keys to bit width for classified result assembly.

    Classical registers use their creg name and width. Loose clbits (not in any
    register) each appear under ``_bit0``, ``_bit1``, … with width ``1``, matching
    :func:`~qiskit_qm_provider.backend.qua_circuit_compilation._loose_bit_keys` and
    QUA stream names in :func:`~qiskit_qm_provider.job.qua_programs.get_run_program`.
    """
    from .qua_circuit_compilation import _loose_bit_keys

    sizes = {creg.name: creg.size for creg in qc.cregs}
    for key in _loose_bit_keys(qc):
        sizes[key] = 1
    return sizes


def experiment_result_header(qc: QuantumCircuit) -> dict[str, Any]:
    """Build a Qiskit :class:`~qiskit.result.models.ExperimentResult` header for *qc*.

    One circuit maps to one experiment result in the legacy ``Result`` API.  The
    header fields mirror reference simulators (e.g. ``BasicSimulator``) so
    :meth:`~qiskit.result.Result.get_counts`, :meth:`~qiskit.result.Result.get_memory`,
    and name-based experiment lookup via :meth:`~qiskit.result.Result.data` work as
    expected.

    ``creg_sizes`` follows :func:`measurement_output_bit_sizes` key order (classical
    registers first, then loose clbits as ``_bitN`` entries of width ``1``).  That
    matches the bit order produced when classified measurement streams are joined in
    :meth:`~qiskit_qm_provider.job.qm_job.QMJob._build_result_function`.
    """
    from .qua_circuit_compilation import _loose_bit_keys

    output_sizes = measurement_output_bit_sizes(qc)
    creg_sizes = [[name, size] for name, size in output_sizes.items()]
    memory_slots = sum(output_sizes.values())

    clbit_labels = [[creg.name, j] for creg in qc.cregs for j in range(creg.size)]
    for key in _loose_bit_keys(qc):
        clbit_labels.append([key, 0])

    return {
        "name": qc.name,
        "n_qubits": qc.num_qubits,
        "qreg_sizes": [[qreg.name, qreg.size] for qreg in qc.qregs],
        "creg_sizes": creg_sizes,
        "qubit_labels": [[qreg.name, j] for qreg in qc.qregs for j in range(qreg.size)],
        "clbit_labels": clbit_labels,
        "memory_slots": memory_slots,
        "global_phase": qc.global_phase,
        "metadata": qc.metadata if qc.metadata is not None else {},
    }


def has_conflicting_calibrations(circuits: List[QuantumCircuit]) -> bool:
    """Check whether circuits define conflicting custom calibrations.

    Args:
        circuits: Circuits whose ``calibrations`` attributes are checked.

    Returns:
        ``True`` if the same operation identifier appears more than once.
    """
    from qm_qasm import OperationIdentifier

    custom_gates = set()
    for qc in circuits:
        if hasattr(qc, "calibrations") and qc.calibrations:
            for gate_name, cal_info in qc.calibrations.items():
                for qubits, parameters in cal_info.keys():
                    op_id = OperationIdentifier(gate_name, len(parameters), qubits)
                    if op_id not in custom_gates:
                        custom_gates.add(op_id)
                    else:
                        return True
    return False


def look_for_standard_op(op: str):
    op = op.lower()
    mapping = {
        "cphase": "cp",
        "cnot": "cx",
        "-x/2": "sxdg",
        "-x90": "sxdg",
        "-y/2": "sydg",
        "-y90": "sydg",
        "x/2": "sx",
        "x90": "sx",
        "x180": "x",
        "y180": "y",
        "y90": "sy",
        "y/2": "sy",
        "hadamard": "h",
        "identity": "id",
        "wait": "delay",
        "readout": "measure",
        "meas": "measure",
        "zz": "rzz",
        "yy": "ryy",
        "xx": "rxx",
    }
    for key in mapping.keys():
        if key in op:
            return mapping[key]
    return mapping.get(op, op)


def get_extended_gate_name_mapping():
    """
    Returns a dictionary of gate names to standard gate instances, with additional custom gates.
    Custom gates are:
    - SYGate: Rotation around the Y axis by π/2
    - SYdgGate: Rotation around the Y axis by -π/2
    - CRGate: Cross-resonance gate
    - FSimGate: Two-qubit gate parametrized by (θ, ϕ)

    Args:
        None

    Returns:
        A dictionary of gate names to standard gate instances, with additional custom gates.
    """
    gate_map = get_standard_gate_name_mapping()
    gate_map["sy"] = SYGate()
    gate_map["cr"] = CRGate()
    gate_map["sydg"] = SYdgGate()
    gate_map["fsim"] = FSimGate(Parameter("θ"), Parameter("ϕ"))

    return gate_map


def has_reset_at_boundary(circuit: QuantumCircuit) -> bool:
    """Check if each qubit in the QuantumCircuit has a reset at the start or end."""
    instructions = circuit.data
    qubits = circuit.qubits

    if not instructions:
        return True  # Empty circuit means all qubits are in reset state

    # Create per-qubit instruction lists
    qubit_instructions = {q: [] for q in qubits}
    for inst in instructions:
        for q in inst.qubits:
            qubit_instructions[q].append(inst)

    # Check each qubit's first and last operations
    for qubit_insts in list(qubit_instructions.values()):
        if not qubit_insts:
            continue  # No instructions means qubit remained in reset state

        # Check first operation on this qubit
        has_start_reset = qubit_insts[0].operation.name == "reset"

        # Check last operation on this qubit
        has_end_reset = qubit_insts[-1].operation.name == "reset"

        if not (has_start_reset or has_end_reset):
            return False

    return True


def binary(val: int, num_bits: int = 0) -> str:
    """Convert an integer to a zero-padded binary string.

    Args:
        val: Integer value to convert.
        num_bits: Minimum width of the binary representation.

    Returns:
        Binary string without the ``0b`` prefix.
    """
    return bin(val)[2:].zfill(num_bits)


def _require_qua_struct_handle(struct: Any) -> Any:
    """Return ``struct`` after validating it is a Quarc :class:`QuaStructHandle`."""
    try:
        from quarc.dsl.structs.qua_struct_handle import QuaStructHandle
    except ImportError as exc:
        raise ImportError("assign_struct_with_table requires the `quarc` package. Install `quarc` to use it.") from exc
    if not isinstance(struct, QuaStructHandle):
        raise TypeError(
            "struct must be a quarc QuaStructHandle (from module.add_struct). " f"Got {type(struct).__name__}."
        )
    return struct


def _struct_field_specs(struct: Any) -> Dict[str, Dict[str, Any]]:
    """Return ``{field_name: {is_array, length}}`` for a QuaStructHandle."""
    from typing import get_args, get_origin, get_type_hints

    from quarc import Array, Scalar

    annotations = get_type_hints(struct._struct_spec.struct)
    specs: Dict[str, Dict[str, Any]] = {}
    for field_name, annotation in annotations.items():
        origin = get_origin(annotation)
        if origin is Scalar:
            specs[field_name] = {"is_array": False, "length": 0}
        elif origin is Array:
            args = get_args(annotation)
            specs[field_name] = {"is_array": True, "length": args[1]}
        else:
            raise TypeError(f"Struct field {field_name!r} has unsupported Quarc annotation {annotation!r}.")
    return specs


def _validate_struct_table_match(struct: Any, table: "ParameterTable") -> None:
    struct_fields = _struct_field_specs(struct)
    table_fields = {parameter.name: parameter for parameter in table.parameters}

    if set(struct_fields) != set(table_fields):
        missing_in_table = sorted(set(struct_fields) - set(table_fields))
        missing_in_struct = sorted(set(table_fields) - set(struct_fields))
        details = []
        if missing_in_table:
            details.append(f"missing from ParameterTable: {missing_in_table}")
        if missing_in_struct:
            details.append(f"missing from struct: {missing_in_struct}")
        raise ValueError(
            "Struct fields and ParameterTable parameters must have exactly the same names. " + "; ".join(details)
        )

    for field_name, spec in struct_fields.items():
        parameter = table_fields[field_name]
        if parameter.is_array != spec["is_array"]:
            raise ValueError(
                f"Field {field_name!r}: struct/table shape mismatch "
                f"(struct is_array={spec['is_array']}, parameter is_array={parameter.is_array})."
            )
        if spec["is_array"] and parameter.length != spec["length"]:
            raise ValueError(
                f"Field {field_name!r}: struct array length {spec['length']} does not match "
                f"parameter length {parameter.length}."
            )
        if not parameter.is_declared:
            raise ValueError(
                f"Parameter {field_name!r} is not declared in QUA. Call "
                f"ParameterTable.declare() (or initialize the owning OPNIC table) first."
            )

    if struct.qua_struct is None:
        raise ValueError("Struct is not initialized in QUA. Call QuaStructHandle.initialize_in_qua() first.")


def assign_struct_with_table(struct: Any, table: "ParameterTable") -> None:
    """QUA macro: copy declared parameter values into a matching OPNIC struct.

    Call inside the same ``with program():`` block after both sides are ready:
    the :class:`~.ParameterTable` parameters must already be declared (via
    :meth:`~.ParameterTable.declare` or OPNIC
    :meth:`~.ParameterTable.initialize_in_qua`), and ``struct`` must be a Quarc
    **QuaStructHandle** returned by ``module.add_struct(...)`` with
    :meth:`QuaStructHandle.initialize_in_qua` already invoked.

    For each field, this macro assigns the table parameter's QUA variable onto
    the corresponding struct field using ``qm.qua.assign``. Field names and
    shapes (scalar vs. array length) must match exactly between the struct spec
    and the table.

    Args:
        struct: A Quarc ``QuaStructHandle`` bound to the destination OPNIC struct.
            The type is validated at runtime via a lazy ``quarc`` import; it is
            not imported for static type checking in this module.
        table: Source :class:`~.ParameterTable` whose declared QUA variables are
            copied field-by-field into ``struct``.

    Raises:
        ImportError: If ``quarc`` is not installed.
        TypeError: If ``struct`` is not a Quarc ``QuaStructHandle``.
        ValueError: If field names or sizes differ, parameters are undeclared,
            or ``struct`` was not initialized in QUA.
    """
    from ..parameter_table import ParameterTable

    if not isinstance(table, ParameterTable):
        raise TypeError(f"table must be a ParameterTable, got {type(table).__name__}.")

    struct = _require_qua_struct_handle(struct)
    _validate_struct_table_match(struct, table)

    for field_name in _struct_field_specs(struct):
        parameter = table.table[field_name]
        struct_field = getattr(struct.qua_struct, field_name)
        if parameter.is_array:
            for index in range(parameter.length):
                assign(struct_field[index], parameter.var[index])
        else:
            assign(struct_field[0], parameter.var)


def _measurement_var_is_array(var) -> bool:
    """Return whether a compiler-wired classical output is a QUA bool array.

    Scalar outputs are bool :class:`~qm.qua._expressions.QuaVariable` instances;
    multi-bit (and some size-``1``) outputs are
    :class:`~qm.qua._expressions.QuaArrayVariable` (or subclasses).
    """
    from qm.qua._expressions import QuaArrayVariable

    return isinstance(var, QuaArrayVariable)


def pack_register_to_int(var, size: int):
    """Pack classical bits into a single integer (LSB = bit index 0).

    Packing follows the wired QUA variable shape: arrays are indexed as ``var[i]``;
    scalars are cast directly (only valid when ``size == 1``).

    Args:
        var: Compiler-owned QUA bool scalar or bool array from ``result_program``.
        size: Number of bits to pack from ``var``.

    Raises:
        ValueError: If ``size`` is not positive, or a multi-bit output is wired to a
            scalar QUA variable.
    """
    if size < 1:
        raise ValueError(f"pack_register_to_int requires size >= 1, got {size}.")

    if _measurement_var_is_array(var):
        return sum(
            (((1 << i) * Cast.to_int(var[i])) for i in range(1, size)),
            start=Cast.to_int(var[0]),
        )

    if size != 1:
        raise ValueError(
            f"Expected a QUA bool array for a {size}-bit classical register, " f"got scalar {type(var).__name__}."
        )
    return Cast.to_int(var)


def get_measurement_outcomes(
    qc: QuantumCircuit, result: CompilationResult, compute_state_int: bool = True
) -> dict[str, dict[str, QuaVariableInt]]:
    """Wire classical measurement outcomes from an embedded circuit into QUA variables.

    Call inside the same ``with program():`` block **immediately after**
    :meth:`~.QMBackend.quantum_circuit_to_qua`. The returned QUA variables reference
    outcomes from the circuit execution that just completed, enabling real-time QUA
    control flow and streaming without round-tripping through Python.

    Args:
        qc: The :class:`~qiskit.circuit.QuantumCircuit` that was compiled.
        result: The compilation result returned by ``quantum_circuit_to_qua``, or a
            :class:`~qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation`
            wrapper.
        compute_state_int: If ``True`` (default), declare an integer packing of
            each register's bits (LSB = qubit index 0).

    Returns:
        A dictionary mapping each output key to a sub-dictionary:

        - ``"value"``: QUA variable for the output (a bool array for multi-bit
          classical registers, a bool scalar for loose clbits).
        - ``"is_array"``: ``True`` when ``"value"`` is a QUA array, ``False`` for a scalar
          — lets callers choose ``value[i]`` vs ``value`` when saving. Mirrors
          ``Parameter.is_array``.
        - ``"length"``: ``Parameter`` convention — ``0`` for a scalar output (loose clbit),
          otherwise the register's bit count.
        - ``"state_int"``: QUA ``int`` with packed bit values, LSB = bit 0 (when
          ``compute_state_int=True``).
        - ``"stream"``: QUA stream for ``stream_processing()`` on the host.

        Each classical register appears under its own name. Loose clbits not in any
        register appear under their own per-bit keys ``_bit0``, ``_bit1``, … (one entry
        per bit, never packed into a single register) — the same keys the
        :attr:`~qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.outputs`
        table exposes. All entries are sourced from that table, so ``state_int`` is the
        cached, rewire-aware handle (``meas[key]["state_int"] ≡ comp.outputs.state_ints[key]``).
    """
    from .qua_circuit_compilation import MeasurementOutcomeTable, QuaCircuitCompilation, _loose_bit_keys

    if isinstance(result, QuaCircuitCompilation):
        table = result.outputs
    else:
        table = MeasurementOutcomeTable.from_compilation(qc, result)

    def _entry(key: str) -> dict:
        field = table.get_parameter(key)
        entry = {
            "value": field.var,
            "is_array": field.is_array,
            "stream": field.stream,
            "length": field.length,
        }
        if compute_state_int:
            entry["state_int"] = field.state_int
        return entry

    clbits_dict: dict[str, dict] = {creg.name: _entry(creg.name) for creg in qc.cregs}
    # Loose clbits are independent single bits, not a register — expose one entry each.
    for key in _loose_bit_keys(qc):
        clbits_dict[key] = _entry(key)

    return clbits_dict


def logically_active_qubits(circuit):
    """
    Retrieve the qubits that are logically active in the circuit, meaning those that carry any operation other than a delay (related to scheduling passes).
    """

    active = set()

    for instr, qargs, _ in circuit.data:
        if instr.name == "delay":
            continue
        for q in qargs:
            active.add(q)

    return sorted(active, key=lambda q: circuit.find_bit(q).index)


def get_non_trivial_observables(observables: PauliList, active_qubit_indices: List[int]) -> PauliList:
    """Restrict observables to logically active qubits.

    Args:
        observables: Pauli observables defined on the full qubit register.
        active_qubit_indices: Indices of qubits that participate in the circuit.

    Returns:
        A :class:`~qiskit.quantum_info.PauliList` with inactive qubits replaced
        by identity.
    """
    new_observables = []
    for observable in observables:
        label = observable.to_label()
        new_pauli_label = ""
        for j in range(observable.num_qubits):
            if j in active_qubit_indices:
                new_pauli_label += label[-j - 1]
        new_observables.append(Pauli(new_pauli_label))

    return PauliList(new_observables)


def get_qua_script(
    backend: "QMBackend",
    circuit: QuantumCircuit,
    param_table=None,
) -> str:
    """Compile a circuit to QUA and return the running QUA script as a string.

    Args:
        backend: The QM backend (used for quantum_circuit_to_qua and generate_config).
        circuit: The transpiled QuantumCircuit to compile to QUA.
        param_table: Optional parameter table for parameterized circuits
            (same as for quantum_circuit_to_qua).

    Returns:
        The QUA script string (Python source of the program that would be executed).
    """
    compilation_result = backend.quantum_circuit_to_qua(circuit, param_table=param_table)
    qua_program = compilation_result.result_program.dsl_program
    config = backend.generate_config()
    return generate_qua_script(qua_program, config)


def dump_qua_script(
    backend: "QMBackend",
    circuit: QuantumCircuit,
    path: str | None = None,
    param_table=None,
) -> str:
    """Compile a circuit to QUA and write the running QUA script to a file.

    Useful for inspecting the actual QUA program that would be executed
    when running the circuit on the backend.

    Args:
        backend: The QM backend (used for quantum_circuit_to_qua and generate_config).
        circuit: The transpiled QuantumCircuit to compile to QUA.
        path: Output file path for the QUA script. If None, uses "debug_qua.py".
        param_table: Optional parameter table for parameterized circuits
            (same as for quantum_circuit_to_qua).

    Returns:
        The path to the written file.
    """
    qua_script = get_qua_script(backend, circuit, param_table=param_table)
    if path is None:
        path = "debug_qua.py"
    with open(path, "w") as f:
        f.write(qua_script)
    return path
