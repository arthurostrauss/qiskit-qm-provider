---
title: Backend and Utilities
nav_order: 4
parent: Home
---

# Backend and Utilities

## QMBackend

`qiskit_qm_provider.backend.qm_backend.QMBackend`

The Qiskit backend implementation for the Quantum Orchestration Platform. It wraps a Quam instance and handles circuit compilation and execution.

### Key Methods

#### `run(run_input: QuantumCircuit | List[QuantumCircuit], **options) -> QMJob`
Executes the circuit(s).
- `run_input`: One or more Qiskit QuantumCircuits.
- `options`: Execution options (e.g., `shots`).

**Generated QUA program (debugging):** `backend.run()` automatically generates the underlying QUA
`Program` that will be executed on QOP. The returned `QMJob` exposes it on `job.program`. To print
it as a QUA script:

```python
from qm import generate_qua_script
print(generate_qua_script(job.program))
```

See
[Workflows and Examples](workflows.md#31-generated-qua-programs-and-how-to-inspect-them)
for a complete snippet showing `backend.run()` and the V2 primitives (`QMSamplerV2`,
`QMEstimatorV2`) side-by-side.

#### `quantum_circuit_to_qua(qc: QuantumCircuit, param_table: Optional[ParameterTable] = None) -> CompilationResult`
Compiles a Qiskit circuit to QUA instructions. Can be used inside a QUA program context.
- `qc`: The circuit to compile.
- `param_table`: Optional `ParameterTable` for mapping Qiskit parameters to real-time QUA variables.

#### `schedule_to_qua_macro(sched: Schedule, param_table: Optional[ParameterTable] = None, input_type: Optional[InputType] = None) -> Callable`
Converts a Qiskit Pulse schedule to a QUA macro.

---

## Utilities

`qiskit_qm_provider.backend.backend_utils`

### `add_basic_macros(backend: QMBackend | QuamRoot, reset_type: Literal["active", "thermalize"] = "thermalize")`
Populates the backend's Quam instance with standard default macros (`x`, `sx`, `rz`, `measure`, `reset`, `delay`, `id`, `cz`).
- `backend`: The backend or machine instance.
- `reset_type`: "active" or "thermalize".

### `get_measurement_outcomes(qc: QuantumCircuit, result: CompilationResult, compute_state_int: bool = True) -> dict`
Extracts measurement outcomes from a compilation result within a QUA program.
- `qc`: The compiled circuit.
- `result`: The result object returned by `quantum_circuit_to_qua`.
- `compute_state_int`: If True (default), computes the integer value of classical registers (`state_int`).

Returns a dictionary:
```python
{
    "creg_name": {
        "value": [qua_var_bit_0, qua_var_bit_1, ...],
        "state_int": qua_var_int,
        "size": int
    }
}
```

---

## QMInstructionProperties

`qiskit_qm_provider.backend.qm_instruction_properties.QMInstructionProperties`

Used in Qiskit 2.x to define custom properties (specifically QUA macros) for instructions in the `Target`.

### `__init__(duration=None, error=None, qua_pulse_macro=None)`
- `qua_pulse_macro`: A callable or QuamMacro that generates the QUA code for the instruction.
