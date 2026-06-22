# Parameter Table

[`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) is the **contract for classical-quantum data** — a single definition shared by Python host logic and QUA program logic, avoiding name and shape mismatches in streaming-heavy workflows.

For method signatures, see the [Parameter Table API reference](apidocs/qm_parameter_table.rst).

## Purpose

Qiskit assumes parameters are bound per circuit run. QUA assumes **long-running programs** with nested loops, streaming, and explicit buffers. Hybrid workflows (feedback, error correction, QUARC/OPNIC control) need a stable mapping between:

- Qiskit circuit parameters and classical input variables, and
- QUA variables loaded from streams, IO, or QUARC-backed OPNIC.

[`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) and [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) provide that mapping.

## Two parameter categories
Both `ParameterTable` and `Parameter` share **the same canonical verb names**, so callers never need to branch on the concrete type:

| Concept | Canonical method | Deprecated alias |
|---|---|---|
| Declare QUA variables | `declare(pause_program=False)` | `declare_variables()` |
| Receive / load input | `rcv(filter_function=None)` | `load_input_values()` |
| Declare output streams | `declare_stream()` | `declare_streams()` |
| Zero-reset QUA variables | `reset_qua()` | `reset_vars()` |

- **`declare(pause_program=False)`**
    Declares all QUA variables in the table. Call this within a QUA program.

1. **Symbolic circuit parameters → real-time QUA variables** — build with [`ParameterTable.from_qiskit()`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) from a circuit's symbolic parameters. Values can be updated in real time (phase, amplitude, frame rotation).
2. **Custom streamed classical data** — define [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) objects directly with [`InputType`](apidocs/stubs/qiskit_qm_provider.parameter_table.InputType.rst) and [`Direction`](apidocs/stubs/qiskit_qm_provider.parameter_table.Direction.rst) for host↔device data (e.g. syndrome integers).

- **`rcv(filter_function=None)`**
    Loads values from the configured input mechanism (Input Stream, IO, OPNIC).

**ASCII parameter names:** QOP rejects non-ASCII names. Use `theta`, `phi`, not Greek symbols, when building tables from Qiskit circuits.

## Lifecycle

**QUA side:**

1. `declare()` — create QUA variables inside `with program():` (works for both `Parameter` and `ParameterTable`).
2. Pass the table to `quantum_circuit_to_qua(qc, param_table=...)`.
3. `rcv()` — read streamed parameters from the host.
4. `stream_back()` — push values to output streams.

**Python side:**

- `push_to_opx(param_dict, job, qm)` — send values to the OPX.
- `fetch_from_opx(job, ...)` — retrieve streamed results.

## Minimal example

```python
from qiskit_qm_provider import Parameter, ParameterTable, Direction, InputType
```

Represents a single parameter. Shares the same canonical interface as `ParameterTable` (see table above).

# Streamed syndrome integer (host ← device)
syndrome_data = Parameter(
    "syndrome_data",
    0,
    input_type=InputType.INPUT_STREAM,
    direction=Direction.INCOMING,
)

**Parameter `__init__` arguments:**
- `name`: Parameter name.
- `value`: Initial Python-side value (also used by `push_to_opx()` when called with no argument).
- `qua_type`: `int`, `float`, `bool`, or `fixed`.
- `input_type`: `InputType` enum.
- `direction`: `Direction` enum (for OPNIC).

# Recovery circuit parameters (host → device)
recovery_vars = ParameterTable.from_qiskit(
    recovery_circuit,
    input_type=InputType.INPUT_STREAM,
)
```

Inside QUA:

- **`declare(pause_program=False)`**: Declares the QUA variable.  *Alias:* `declare_variable()`.
- **`rcv()`**: Loads one value from the input mechanism.  *Alias:* `load_input_value()`.
- **`assign(value)`**: Assigns a value (can be QUA variable or constant) to this parameter.
- **`save_to_stream()`**: Saves the parameter's current value to its declared stream.
- **`push_to_opx(value=self.value, ...)`**: Sends a value to the OPX. When called with no positional argument, uses `self.value` — set it once and call `push_to_opx()` without repeating the value.

```python
recovery_vars.declare()
syndrome_data.declare()
syndrome_data.declare_stream()
```

For the full error-correction loop using these tables, see [Error-Correction Workflow](error_correction.md).

## ParameterPool

`ParameterPool` manages unique ids, the global registry of all `Parameter` and `ParameterTable` objects, and the single bound Quarc `BaseModule` slot.

### 1-field struct promotion in `from_quarc_module()`

When deserializing a Quarc module via `ParameterPool.from_quarc_module(module)`, any struct with **exactly one field** is automatically promoted to a standalone `Parameter` instead of a `ParameterTable`. The wrapper `ParameterTable` is still created internally and marked as *synthetic-standalone*, so OPNIC transport (`declare`, `rcv`, `push_to_opx`) delegates through it correctly.

```python
tables = ParameterPool.from_quarc_module(module)
# Scalar quantities (1 field) → Parameter
n_shots = tables["n_shots"]         # Parameter, not ParameterTable
n_shots.declare()                   # works — delegates via synthetic table
n_shots.rcv()                       # works — calls table._var.recv()
n_shots.value = 1000
n_shots.push_to_opx()              # streams n_shots.value = 1000

# Multi-field quantities → ParameterTable (unchanged)
policy = tables["policy_params"]    # ParameterTable
policy.declare()
policy.rcv()
```

This eliminates any manual downconversion of 1-element `ParameterTable` objects back to `Parameter` on the deserializing side.

### Two pipelines

- **Pipeline 1 — module-first:** call `ParameterPool.from_quarc_module(my_module)` with a pre-built `BaseModule`. The pool wraps each struct as a `ParameterTable` (or promotes to `Parameter` for 1-field structs) and binds `my_module` as the pool's accumulator.
- **Pipeline 2 — parameters-first:** declare `Parameter`/`ParameterTable` objects with `input_type=OPNIC`, then call `ParameterPool.to_quarc_module()` to flush all pending tables onto a freshly-created `BaseModule`.

---

## InputType & Direction

## Supporting types

### `InputType`
Enum for input mechanisms:
- `OPNIC`: OPNIC (classical host) packet communication.
- `INPUT_STREAM`: Standard QOP Input Stream.
- `IO1`, `IO2`: GPIO inputs.
- **[`ParameterPool`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterPool.rst)** — coordinate multiple tables in one program.
- **[`QUA2DArray`](apidocs/stubs/qiskit_qm_provider.parameter_table.QUA2DArray.rst) / `QUAArray`** — multi-index parameter memory (flattened QUA arrays).
- **`ParameterVector` note:** OpenQASM 3 exports `ParameterVector` elements as individual parameters; `from_qiskit` handles this transparently.

## Related

- **Guide:** [Measurement outputs](measurement_outputs.md), [Backend — embedding](backend.md#embed-in-qua-hybrid), [Error correction](error_correction.md)
- **API:** [Parameter Table reference](apidocs/qm_parameter_table.rst), [Backend — measurement outputs](apidocs/qm_backend.rst#circuit-compilation-and-measurement-outputs)
- **Workflows:** [Hybrid embedding](workflows.md#4-hybrid-quaqiskit-programs-embedding-circuits-in-qua)
