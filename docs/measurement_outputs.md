# Measurement outputs (`comp.outputs`)

Compiled-circuit measurement handles are **compilation-local**: they wire classical register (and loose clbit) outcomes from qm-qasm's `result_program` into QUA variables. They are **not** runtime/OPNIC deployment parameters.

**API reference (autodoc):** [`QuaCircuitCompilation`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.rst), [`MeasurementOutcomeTable`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable.rst), [`MeasurementRegisterField`](apidocs/stubs/qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField.rst), [`QuaFieldTable`](apidocs/stubs/qiskit_qm_provider.parameter_table._mixins.QuaFieldTable.rst), and [scope guards](apidocs/qm_parameter_table.rst#qua-program-scope-guards).

See also: [Workflows — hybrid programs](workflows.md#4-hybrid-quaqiskit-programs-embedding-circuits-in-qua).

## Locality model

| Concept | Runtime `ParameterTable` | `comp.outputs` (`MeasurementOutcomeTable`) |
|---------|--------------------------|--------------------------------------------|
| **Scope** | Process/session; registered in `ParameterPool` (runtime registry) | **Per `QuaCircuitCompilation`**; weakref-tracked, not OPNIC-emitted |
| **Purpose** | Host↔OPX knobs, OPNIC structs, input streams | **Compiler output handles** wired from `result_program` |
| **Typical classical path** | `stream_back()` / `fetch_from_opx()` on struct/table | `save(state_int, stream)` — stream processing on host |
| **Name keys** | User-chosen struct/table/field names | **Circuit output keys** (creg names + `_bitN` loose bits) |

**Key guidance:** matching an OPNIC struct field name to a creg name is optional and usually unnecessary. Same string is allowed (dual namespace) but does not imply the same QUA variable. Use `ParameterPool.lookup_runtime_parameter(name)` for runtime knobs; use `comp.outputs.get_parameter(name)` for measurement fields.

## Accessor contract (aligned with `ParameterTable`)

All QUA variable accessors require `with program():`.

| Access | Returns |
|--------|---------|
| `comp.outputs["c"]` / `comp.outputs.c` | QUA bool var or array (measurement outcome) |
| `comp.outputs.get_parameter("c")` | `MeasurementRegisterField` handle |
| `comp.outputs.get_variable("c")` | QUA var (same as `["c"]`) |
| `comp.outputs.state_ints["c"]` | Lazy-packed `int` QUA scalar |
| `comp.outputs.streams["c"]` | Per-field `declare_stream()` handle |

**Breaking change:** `comp.outputs.c.state_int` is invalid — `comp.outputs.c` is the measurement bool var. Use `comp.outputs.state_ints["c"]` or `comp.outputs.get_parameter("c").state_int`.

## Worked examples

### RL reward stream

```python
from qm.qua import program, save

with program() as prog:
    comp = backend.quantum_circuit_to_qua(reward_circuit)
    save(comp.outputs.state_ints["meas"], comp.outputs.streams["meas"])
```

Host-side RL typically consumes the packed integer stream, not raw bool arrays. Input struct fields (actions) and output streams (rewards) are separate pipelines.

### QEC — in-QUA processing first

For error correction, the OPNIC struct you `stream_back()` is usually **not** the same object as the raw measurement register from the circuit. Compiler outputs (`comp.outputs`) hold discriminated bits from `result_program`; runtime [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) fields hold whatever **derived** classical data the host decoder needs — detection events, packed syndrome integers, histogram bins, etc. See the [Error-correction guide](error_correction.md) for full walkthroughs.

**Minimal pattern — derive before streaming:**

```python
from qm.qua import program, assign, declare

with program() as prog:
    comp = backend.quantum_circuit_to_qua(syndrome_circuit)
    syndrome = comp.outputs["syndrome"]          # bool array var — raw readout
    detection_events = declare(int)
    # ... QUA logic combining syndrome bits into detection_events ...
    save(detection_events, comp.outputs.streams["syndrome"])
```

**Recommended pattern — detection events via consecutive-round XOR:**

Many decoders expect per-bit **changes** between rounds, not absolute stabilizer values. Declare runtime tables for staging and streaming, XOR in QUA, and stream only the derived table:

```python
from qm.qua import declare, for_, assign

def update_syndrome_streams(
    circuit,
    comp,
    previous_measurement_outcomes: ParameterTable,  # local history table
    syndrome_data: ParameterTable,  # OPNIC / stream transport
):
    """Update syndrome streams for a given circuit."""
    j = declare(int)
    for creg in circuit.cregs:
        meas_reg = comp.outputs[creg.name]
        syndrome_param = syndrome_data[creg.name]
        prev_meas = previous_measurement_outcomes[creg.name]
        with for_(j, 0, j < creg.size, j + 1):
            assign(
                syndrome_param[j],
                prev_meas[j] ^ meas_reg[j],
            )
            assign(prev_meas[j], meas_reg[j])
    syndrome_data.stream_back(reset=True)
```

After each `comp = backend.quantum_circuit_to_qua(syndrome_circuit)`, pass `comp` to `update_syndrome_streams(...)`. The host receives detection events from `syndrome_data`, not raw `comp.outputs` bits.

**Large registers — prefer `state_int` for streaming:**

For registers with many bits, avoid buffering full bool chains on the stream path. Use the lazy-packed integer instead:

```python
from qm.qua import program, assign

with program() as prog:
    comp = backend.quantum_circuit_to_qua(syndrome_circuit)

    ancilla = syndrome_circuit.cregs[0].name
    assign(syndrome_data.var, comp.outputs.state_ints[ancilla])
    syndrome_data.stream_back(reset=True)
```

`state_int` collapses `creg.size` bits into one scalar — useful for host decoders, lookup tables, and `stream_processing()` buffers of size `2**creg.size`. Keep per-bit bool access when you need XOR or single-stabilizer feedback; switch to `state_int` when the outcome is consumed as a single label. Details: [Error-correction — detection events and state_int](error_correction.md#detection-events--consecutive-round-xor).

QEC workflows process syndrome bits in QUA before streaming a derived quantity — no 1:1 creg→OPNIC struct mapping is required.

### Optional bridge to OPNIC (explicit only)

When the host must receive data via an OPNIC struct rather than a raw stream:

```python
with program() as prog:
    comp = backend.quantum_circuit_to_qua(qc)
    reward_table.assign({"detection_events": comp.outputs.state_ints["syndrome"]})
    reward_table.stream_back()
```

Never automatic by name — always an explicit `assign`.

## Re-compile and lifecycle

- Each `quantum_circuit_to_qua` call creates a new `QuaCircuitCompilation` with **fresh** `MeasurementRegisterField` objects.
- `comp.rewire_outputs(qc, new_result)` refreshes wiring on the same wrapper; size or compilation identity changes invalidate cached `state_int` / `stream` handles.
- `ParameterPool.reset()` clears weakref measurement registries together with the runtime registry.
- Debug introspection: `ParameterPool.iter_measurement_outcome_tables()`, `ParameterPool.iter_measurement_register_fields()`.

## Future extensibility

Today, output keys mirror classical registers (and loose clbits) from `result_program`. qm-qasm may later expose non-creg output vars; `comp.outputs` will wire whatever keys the compiler exposes. Keys come from the compilation result, not from user struct definitions.

## Edge-case matrix (summary)

| Issue | Status |
|-------|--------|
| `table["c"]` vs `get_parameter("c")` semantics | **Solved** — vars vs handles |
| Global dedup / stale size across compiles | **Solved** — per-compilation fields + rewire invalidation |
| `isinstance(x, Parameter)` for measurements | **Solved** — `MeasurementRegisterField` is not a `Parameter` |
| Same name runtime + measurement | **Allowed** — dual namespace; use role-specific accessors |
| OPNIC field name = creg name | **Doc only** — allowed but usually unnecessary |
| `deepcopy` measurement field | **Raises** `TypeError` |

Legacy [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst) remains available; it uses `get_parameter()` internally and accepts `QuaCircuitCompilation` or raw `CompilationResult`.
