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

Standard Qiskit backend workflow: compile, execute, return a [`QMJob`](apidocs/stubs/qiskit_qm_provider.job.QMJob.rst). The generated QUA program is on `job.program`:

```python
from qm import generate_qua_script

job = backend.run(qc, shots=256)
print(generate_qua_script(job.program))
result = job.result()
```

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

Returns `dict[creg_name, subdict]` — one entry per classical register in the circuit (plus a synthetic `_bit` register for loose clbits).

| Key | Role |
|-----|------|
| **`value`** | List of QUA variables — one per bit — holding discriminated 0/1 outcomes from the embedded circuit. Use for bit-level QUA logic. |
| **`size`** | Python `int`: number of bits in the register. |
| **`state_int`** | QUA `int` (when `compute_state_int=True`, the default): integer packing of all bits; **LSB = qubit index 0** (Qiskit convention). Use for compact syndromes or lookup-table indexing. |
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
