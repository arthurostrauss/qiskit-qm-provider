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

- **`declare_variables(pause_program=False)`**
    Declares all QUA variables in the table. Call this within a QUA program.

- **`load_input_values(filter_function=None)`**
    Loads values from the configured input mechanism (Input Stream, IO, DGX).

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

Represents a single parameter.

### `__init__(name, value, qua_type, input_type, direction, units)`
- `name`: Parameter name.
- `value`: Initial value.
- `qua_type`: `int`, `float`, `bool`, or `fixed`.
- `input_type`: `InputType` enum.
- `direction`: `Direction` enum (for DGX).

### Methods

- **`assign(value)`**: Assigns a value (can be QUA variable or constant) to this parameter.
- **`save_to_stream()`**: Saves the parameter's current value to its declared stream.

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
