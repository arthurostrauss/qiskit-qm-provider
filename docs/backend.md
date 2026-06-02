# Backend and Utilities

`QMBackend` is the **central bridge object**: it represents hardware to Qiskit (via a QuAM-derived `Target`) and translates Qiskit artifacts to QUA (via qm_qasm and `quantum_circuit_to_qua`).

For full signatures, see the [Backend API reference](apidocs/qm_backend.rst).

## Purpose

Backends serve two roles:

1. **Represent hardware in Qiskit** — macros from QuAM populate the `Target`; coupling map from topology; qubit properties (T1, T2, frequencies).
2. **Translate circuits to QUA** — OpenQASM 3 export + qm_qasm compilation, or Pulse schedule conversion (Qiskit 1.x legacy).

`FluxTunableTransmonBackend` is the reference implementation: `init_macro` from QuAM's `initialize_qpu`, and QuAM ↔ Pulse channel mapping (`get_quam_channel`, `get_pulse_channel`).

## Two execution modes

### Submit-and-run

Standard Qiskit backend workflow: compile, execute, return a `QMJob`. The generated QUA program is on `job.program`:

```python
from qm import generate_qua_script

job = backend.run(qc, shots=256)
print(generate_qua_script(job.program))
result = job.result()
```

### Embed-in-QUA (hybrid)

Inside `with program():`, compile a circuit as a QUA macro and wire classical outcomes **in the same program**:

```python
from qm.qua import program
from qiskit_qm_provider.backend.backend_utils import get_measurement_outcomes

with program() as prog:
    result = backend.quantum_circuit_to_qua(qc, param_table=my_param_table)
    meas = get_measurement_outcomes(qc, result)
    syndrome_int = meas[creg.name]["state_int"]
    # use syndrome_int immediately in QUA control flow or streaming
```

Call `get_measurement_outcomes` **immediately after** `quantum_circuit_to_qua` in the same QUA program — not as a separate Python post-processing step. The returned variables reference outcomes from the circuit execution that just ran.

## `get_measurement_outcomes` return dictionary

Returns `dict[creg_name, subdict]` — one entry per classical register in the circuit (plus a synthetic `_bit` register for loose clbits).

| Key | Role |
|-----|------|
| **`value`** | List of QUA variables — one per bit — holding discriminated 0/1 outcomes from the embedded circuit. Use for bit-level QUA logic. |
| **`size`** | Python `int`: number of bits in the register. |
| **`state_int`** | QUA `int` (when `compute_state_int=True`, the default): integer packing of all bits; **LSB = qubit index 0** (Qiskit convention). Use for compact syndromes or lookup-table indexing. |
| **`stream`** | QUA stream object for `stream_processing()` — buffer outcomes to the host. |

See [Parameter Table](parameter_table.md) for `stream_back` / `fetch_from_opx` on the Python side.

## Real-time parameters

Unlike typical Qiskit backends, parameters need not be bound at compile time. Use `ParameterTable.from_qiskit()` to map circuit parameters to real-time QUA variables (phases, amplitudes, frame rotations).

**Warning:** QOP rejects non-ASCII parameter names. Use ASCII names (`theta`, `phi`) — not Greek symbols — so OpenQASM 3 and QUA compilation succeed.

## Utilities

### `add_basic_macros`

Seeds standard gate macros on a QuAM machine. **Flux-tunable transmon defaults** tied to `FluxTunableQuam` — see [Providers guide](providers.md#seeding-gate-macros-with-add-basic-macros). Override freely for other platforms.

### `get_qua_script` / `dump_qua_script`

Debug helpers to inspect generated QUA from compilation results.

## Custom calibrations (Qiskit 2.x)

Attach QUA macros to Target instructions via `QMInstructionProperties`:

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

When `QISKIT_PULSE_AVAILABLE`, `schedule_to_qua_macro` converts **gate pulse schedules** to QUA.

| Supported | Not supported |
|-----------|---------------|
| Gate operations as Pulse schedules | Qiskit Pulse **`Measure` / measurement instructions** |
| QuAM ↔ Pulse channel mapping on `FluxTunableTransmonBackend` | Kerneled / raw IQ readout (see [classified-only note](index.md#classified-measurement-outcomes-only)) |

Hybrid readout: circuit-level `measure` → `quantum_circuit_to_qua` → `get_measurement_outcomes`.

## Related

- **Guide:** [Workflows — calibrations](workflows.md#calibrations-and-custom-gates), [hybrid embedding](workflows.md#hybrid-qua-qiskit-programs-embedding-circuits-in-qua)
- **API:** [Backend reference](apidocs/qm_backend.rst), [Pulse reference](apidocs/qm_pulse.rst)
- **Examples:** `examples/circuit_calibrations_pulse.py`, `examples/sampler_workflow.py`
