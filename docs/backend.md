# Backend and Utilities

[`QMBackend`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) is the **central bridge object**: it represents hardware to Qiskit (via a QuAM-derived `Target`) and translates Qiskit artifacts to QUA (via qm_qasm and [`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst)).

For full signatures, see the [Backend API reference](apidocs/qm_backend.rst).

## Purpose

Backends serve two roles:

1. **Represent hardware in Qiskit** — macros from QuAM populate the `Target`; coupling map from topology; qubit properties (T1, T2, frequencies).
2. **Translate circuits to QUA** — OpenQASM 3 export + qm_qasm compilation, or Pulse schedule conversion (Qiskit 1.x legacy).

[`FluxTunableTransmonBackend`](apidocs/stubs/qiskit_qm_provider.backend.FluxTunableTransmonBackend.rst) is the reference implementation: `init_macro` from QuAM's `initialize_qpu`, and QuAM ↔ Pulse channel mapping (`get_quam_channel`, `get_pulse_channel`).

## Two execution modes

### Submit-and-run

Standard Qiskit backend workflow: compile, execute, return a [`QMJob`](apidocs/stubs/qiskit_qm_provider.job.QMJob.rst). The compiled QUA programs are on `job.programs`:

```python
from qm import generate_qua_script

job = backend.run(qc, shots=256)
print(generate_qua_script(job.programs[0]))
result = job.result()
```

### Multi-circuit batches and `max_circuits`

When you pass a **list** of circuits to [`QMBackend.run()`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst), the provider may build **more than one** QUA program and queue them sequentially on QOP. Results are still returned as a **single** Qiskit [`Result`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.result.Result) with experiments in the same order as your input list.

This behaviour also applies to [`QMSamplerV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerV2.rst) and [`QMEstimatorV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMEstimatorV2.rst) — see the [Primitives guide](primitives.md#large-pub-batches-and-max_circuits) for details. It does **not** apply to [`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst).

**Backend option** [`max_circuits`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) (default `30`):

| Value | Effect |
|-------|--------|
| Positive integer | Pack at most this many circuits (or PUBs) per QUA program. Larger batches are split into consecutive chunks (e.g. 70 circuits → 30 + 30 + 10). |
| `None` | Disable size-based splitting — always one QUA program for the full batch (unless calibrations force a split; see below). |

Set it at construction time or update it at any point before submitting a job:

```python
# At construction time:
backend = FluxTunableTransmonBackend(machine, max_circuits=10)

# Or update after construction — applies to the next backend.run() / sampler.run() call:
backend.set_options(max_circuits=10)
```

**Splitting rules** (in priority order):

1. **Conflicting calibrations** — one circuit per QUA program (unchanged).
2. **`len(circuits) > max_circuits`** — consecutive groups of `max_circuits` circuits.
3. **Otherwise** — a single program holding all circuits.

**Inspecting generated QUA** — `job.programs` is always a `list[Program]` (length 1 when no chunking occurred, otherwise one entry per chunk):

```python
from qm import generate_qua_script

circuits = [make_circuit(i) for i in range(100)]
job = backend.run(circuits, shots=256)

for chunk_idx, prog in enumerate(job.programs):
    print(f"=== QUA program {chunk_idx} ===")
    print(generate_qua_script(prog))

result = job.result()  # one Result; experiment order matches `circuits`
```

Use a smaller `max_circuits` when a single packed program grows too large for QOP compilation or device limits. Use `max_circuits=1` to force one circuit per QUA program.

### Embed-in-QUA (hybrid)

Inside `with program():`, compile a circuit as a QUA macro and wire classical outcomes **in the same program**:

```python
from qm.qua import program, save

with program() as prog:
    comp = backend.quantum_circuit_to_qua(qc, param_table=my_param_table)
    save(comp.outputs.state_ints["meas"], comp.outputs.streams["meas"])
```

[`QuaCircuitCompilation`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.rst) exposes wired handles via [`comp.outputs`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable.rst) ([`MeasurementRegisterField`](apidocs/stubs/qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField.rst) per creg). See the [Measurement outputs guide](measurement_outputs.md) for the accessor contract and locality model.

Legacy shim (dict API):

```python
from qiskit_qm_provider.backend.backend_utils import get_measurement_outcomes

with program() as prog:
    comp = backend.quantum_circuit_to_qua(qc, param_table=my_param_table)
    meas = get_measurement_outcomes(qc, comp)
    syndrome_int = meas[creg.name]["state_int"]
```

Call measurement wiring **immediately after** `quantum_circuit_to_qua` in the same QUA program — not as a separate Python post-processing step.

## [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst) return dictionary

Returns `dict[key, subdict]` — one entry per classical register in the circuit, plus one entry per loose clbit under its own key `_bit0`, `_bit1`, … (loose bits are independent single bits, never packed into a single register). Every entry is sourced from `comp.outputs`, so `meas[key]["state_int"]` is exactly `comp.outputs.state_ints[key]`.

| Key | Role |
|-----|------|
| **`value`** | The QUA variable holding discriminated 0/1 outcomes from the embedded circuit — a bool **array** for a multi-bit register, a bool **scalar** for a loose clbit. Use for bit-level QUA logic. |
| **`is_array`** | Python `bool`: `True` when `value` is a QUA array, `False` when it is a scalar — pick `value[i]` vs `value` when saving. Mirrors `Parameter.is_array`. |
| **`length`** | Python `int`, `Parameter` convention: `0` for a scalar output (loose clbit), otherwise the register's bit count. |
| **`state_int`** | QUA `int` (when `compute_state_int=True`, the default): integer packing of all bits; **LSB = qubit index 0** (Qiskit convention). Lazily declared and cached on the underlying field. Use for compact syndromes or lookup-table indexing. |
| **`stream`** | QUA stream object for `stream_processing()` — buffer outcomes to the host. |

See [Parameter Table](parameter_table.md) for `stream_back` / `fetch_from_opx` on the Python side.

## Real-time parameters

Unlike typical Qiskit backends, parameters need not be bound at compile time. Use [`ParameterTable.from_qiskit()`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) to map circuit parameters to real-time QUA variables (phases, amplitudes, frame rotations).

**Warning:** QOP rejects non-ASCII parameter names. Use ASCII names (`theta`, `phi`) — not Greek symbols — so OpenQASM 3 and QUA compilation succeed.

## Utilities

### [`add_basic_macros`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.add_basic_macros.rst)

Seeds standard gate macros on a QuAM machine. **Flux-tunable transmon defaults** tied to `FluxTunableQuam` — see [Providers guide](providers.md#seeding-gate-macros-with-add-basic-macros). Override freely for other platforms.

### `get_qua_script` / `dump_qua_script`

Debug helpers to inspect generated QUA from compilation results.

### [`assign_struct_with_table`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.assign_struct_with_table.rst)

**QUA macro** for OPNIC struct assignment when a :class:`~.ParameterTable` and a Quarc ``QuaStructHandle`` share the same field layout. Call inside ``with program():`` after:

1. ``table.declare()`` (or OPNIC ``table.initialize_in_qua()``) — source parameters must have QUA variables.
2. ``struct.initialize_in_qua()`` — destination struct must be declared in the same program.

The macro copies each table parameter's QUA variable into the matching struct field via ``qm.qua.assign``. Field names and sizes (scalar vs. array length) must match exactly.

```python
from qm.qua import program
from qiskit_qm_provider.backend.backend_utils import assign_struct_with_table

with program() as prog:
    policy_table.declare()
    outbound_handle.initialize_in_qua()
    assign_struct_with_table(outbound_handle, policy_table)
    outbound_handle.send()
```

``struct`` must be a Quarc **QuaStructHandle** from ``module.add_struct(...)`` (validated at runtime via lazy ``quarc`` import).

## Custom calibrations (Qiskit 2.x)

Attach QUA macros to Target instructions via [`QMInstructionProperties`](apidocs/stubs/qiskit_qm_provider.backend.QMInstructionProperties.rst):

```python
from qiskit.circuit import Parameter as QiskitParameter, Gate
from qiskit_qm_provider import QMInstructionProperties

theta = QiskitParameter("theta")
cx_cal = Gate("cx_cal", num_qubits=2, params=[theta])

def qua_macro(theta_val):
    qubit_pair = backend.get_qubit_pair((0, 1))
    qubit_pair.apply("cz", amplitude_scale=theta_val)

properties = {
    (0, 1): QMInstructionProperties(
        duration=backend.target["cx"][(0, 1)].duration,
        qua_pulse_macro=qua_macro,
    )
}
backend.target.add_instruction(cx_cal, properties=properties)
backend.update_target()  # mandatory — syncs qm_qasm with the Target
```

Whenever you modify `backend.target`, call `update_target` so both transpilation and `quantum_circuit_to_qua` see the same gate set.

## Pulse support (Qiskit 1.x legacy)

When `QISKIT_PULSE_AVAILABLE`, [`schedule_to_qua_macro`](apidocs/stubs/qiskit_qm_provider.pulse.schedule_to_qua_macro.rst) converts **gate pulse schedules** to QUA.

| Supported | Not supported |
|-----------|---------------|
| Gate operations as Pulse schedules | Qiskit Pulse **`Measure` / measurement instructions** |
| QuAM ↔ Pulse channel mapping on `FluxTunableTransmonBackend` | Kerneled / raw IQ readout (see [classified-only note](index.md#classified-measurement-outcomes-only)) |

Hybrid readout: circuit-level `measure` → [`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) → [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst).

## Related

- **Guide:** [Workflows — calibrations](workflows.md#calibrations-and-custom-gates), [hybrid embedding](workflows.md#hybrid-qua-qiskit-programs-embedding-circuits-in-qua)
- **API:** [Backend reference](apidocs/qm_backend.rst), [Pulse reference](apidocs/qm_pulse.rst)
- **Examples:** `examples/circuit_calibrations_pulse.py`, `examples/sampler_workflow.py`
