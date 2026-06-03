# Parameter Table

[`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) is the **contract for classical-quantum data** — a single definition shared by Python host logic and QUA program logic, avoiding name and shape mismatches in streaming-heavy workflows.

For method signatures, see the [Parameter Table API reference](apidocs/qm_parameter_table.rst).

## Purpose

Qiskit assumes parameters are bound per circuit run. QUA assumes **long-running programs** with nested loops, streaming, and explicit buffers. Hybrid workflows (feedback, error correction, DGX control) need a stable mapping between:

- Qiskit circuit parameters and classical input variables, and
- QUA variables loaded from streams, IO, or DGX.

[`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) and [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) provide that mapping.

## Two parameter categories

1. **Symbolic circuit parameters → real-time QUA variables** — build with [`ParameterTable.from_qiskit()`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) from a circuit's symbolic parameters. Values can be updated in real time (phase, amplitude, frame rotation).
2. **Custom streamed classical data** — define [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) objects directly with [`InputType`](apidocs/stubs/qiskit_qm_provider.parameter_table.InputType.rst) and [`Direction`](apidocs/stubs/qiskit_qm_provider.parameter_table.Direction.rst) for host↔device data (e.g. syndrome integers).

**ASCII parameter names:** QOP rejects non-ASCII names. Use `theta`, `phi`, not Greek symbols, when building tables from Qiskit circuits.

## Lifecycle

**QUA side:**

1. `declare_variables()` / `declare_variable()` — create QUA variables inside `with program():`.
2. Pass the table to `quantum_circuit_to_qua(qc, param_table=...)`.
3. `load_input_values()` — read streamed parameters from the host.
4. `stream_back()` — push values to output streams.

**Python side:**

- `push_to_opx(param_dict, job, qm)` — send values to the OPX.
- `fetch_from_opx(job, ...)` — retrieve streamed results.

## Minimal example

```python
from qiskit_qm_provider import Parameter, ParameterTable, Direction, InputType

# Streamed syndrome integer (host ← device)
syndrome_data = Parameter(
    "syndrome_data",
    0,
    input_type=InputType.INPUT_STREAM,
    direction=Direction.INCOMING,
)

# Recovery circuit parameters (host → device)
recovery_vars = ParameterTable.from_qiskit(
    recovery_circuit,
    input_type=InputType.INPUT_STREAM,
)
```

Inside QUA:

```python
recovery_vars.declare_variables()
syndrome_data.declare_variable()
syndrome_data.declare_stream()
```

For the full error-correction loop using these tables, see [Error-Correction Workflow](error_correction.md).

## Supporting types

- **[`ParameterPool`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterPool.rst)** — coordinate multiple tables in one program.
- **[`QUA2DArray`](apidocs/stubs/qiskit_qm_provider.parameter_table.QUA2DArray.rst) / `QUAArray`** — multi-index parameter memory (flattened QUA arrays).
- **`ParameterVector` note:** OpenQASM 3 exports `ParameterVector` elements as individual parameters; `from_qiskit` handles this transparently.

## Related

- **Guide:** [Backend — embedding](backend.md#embed-in-qua-hybrid), [Error correction](error_correction.md)
- **API:** [Parameter Table reference](apidocs/qm_parameter_table.rst)
- **Workflows:** [Hybrid embedding](workflows.md#hybrid-qua-qiskit-programs-embedding-circuits-in-qua)
