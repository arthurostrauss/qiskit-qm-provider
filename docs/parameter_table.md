---
title: Parameter Table
nav_order: 6
parent: Home
---

# Parameter Table

The `ParameterTable` module enables real-time parameter updates and streaming.

## ParameterTable

`qiskit_qm_provider.parameter_table.parameter_table.ParameterTable`

Class enabling the mapping of parameters to be updated to their corresponding "to-be-declared" QUA variables.

### `__init__(parameters_dict, name: Optional[str] = None)`
- `parameters_dict`: Can be:
    - A dictionary of form `{ "name": (initial_value, qua_type, input_type, direction) }`.
    - A list of `Parameter` objects.
- `name`: Optional table name.

### Methods

Both `ParameterTable` and `Parameter` share **the same canonical verb names**, so callers never need to branch on the concrete type:

| Concept | Canonical method | Deprecated alias |
|---|---|---|
| Declare QUA variables | `declare(pause_program=False)` | `declare_variables()` |
| Receive / load input | `rcv(filter_function=None)` | `load_input_values()` |
| Declare output streams | `declare_stream()` | `declare_streams()` |
| Zero-reset QUA variables | `reset_qua()` | `reset_vars()` |

- **`declare(pause_program=False)`**
    Declares all QUA variables in the table. Call this within a QUA program.

- **`rcv(filter_function=None)`**
    Loads values from the configured input mechanism (Input Stream, IO, OPNIC).

- **`push_to_opx(param_dict, job, qm, verbosity)`**
    (Client-side) Pushes values to the OPX.
    - `param_dict`: Dictionary of `{parameter_name: value}`.

- **`fetch_from_opx(job, fetching_index, fetching_size)`**
    (Client-side) Fetches values from the OPX (if configured for output/streaming).

- **`stream_back(reset=False)`**
    (QUA-side) Streams the current parameter values back to the client.

- **`from_qiskit(qc: QuantumCircuit, input_type, filter_function)`**
    Class method to generate a table from a Qiskit circuit's parameters.

---

## Parameter

`qiskit_qm_provider.parameter_table.parameter.Parameter`

Represents a single parameter. Shares the same canonical interface as `ParameterTable` (see table above).

### `__init__(name, value, qua_type, input_type, direction, units)`
- `name`: Parameter name.
- `value`: Initial Python-side value (also used by `push_to_opx()` when called with no argument).
- `qua_type`: `int`, `float`, `bool`, or `fixed`.
- `input_type`: `InputType` enum.
- `direction`: `Direction` enum (for OPNIC).

### Methods

- **`declare(pause_program=False)`**: Declares the QUA variable.  *Alias:* `declare_variable()`.
- **`rcv()`**: Loads one value from the input mechanism.  *Alias:* `load_input_value()`.
- **`assign(value)`**: Assigns a value (can be QUA variable or constant) to this parameter.
- **`save_to_stream()`**: Saves the parameter's current value to its declared stream.
- **`push_to_opx(value=self.value, ...)`**: Sends a value to the OPX. When called with no positional argument, uses `self.value` — set it once and call `push_to_opx()` without repeating the value.

---

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

`qiskit_qm_provider.parameter_table.input_type`

### `InputType`
Enum for input mechanisms:
- `OPNIC`: OPNIC (classical host) packet communication.
- `INPUT_STREAM`: Standard QOP Input Stream.
- `IO1`, `IO2`: GPIO inputs.

### `Direction`
Enum for DGX data flow:
- `INCOMING`: OPX -> DGX.
- `OUTGOING`: DGX -> OPX.
- `BOTH`: Bidirectional.
