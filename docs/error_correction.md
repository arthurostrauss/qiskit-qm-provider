# Quantum Error-Correction Workflow

Quantum Error correction is the **canonical stress test** for hybrid Qiskit-in-QUA design: repeated syndrome measurement, classical decoding, and recovery inside a long-running QUA program while circuits stay authored in Qiskit.

For method signatures, see the [Parameter Table API reference](apidocs/qm_parameter_table.rst) and [Backend API reference](apidocs/qm_backend.rst).

## Purpose

Plain circuit submission breaks down for QEC because you need:

- **Many cycles** of measure → decode → recover, not one shot per circuit variant.
- **Streaming classical data** (syndrome integers, detection events) between QUA and a host decoder.
- **Stable parameter names** across Qiskit recovery circuits and QUA variables.

### Why the QEC loop belongs in QUA

Qiskit exposes a real-time [`ForLoopOp`](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.ForLoopOp) that can compile a fixed number of cycles into a native QUA `for_` loop. It does **not**, however, expose a way to stream intermediate measurement outcomes to the host or to stream-processing within that loop. That leaves two unattractive options:

| Approach | Problem |
|----------|---------|
| Reuse the same classical bit each cycle | Intermediate outcomes are overwritten; only the last round is retained |
| Declare one classical variable per cycle | Outcomes are preserved, but memory use scales with cycle count |

The recommended pattern is to author a **single template cycle** (syndrome measure, optional recovery) as a Qiskit circuit, embed it in a QUA `for_` loop, and stream or derive classical data each iteration via [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) — not to unroll the full QEC schedule as one giant Qiskit circuit.

[`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) and [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) make that classical-quantum contract explicit. [`QuaCircuitCompilation.outputs`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.rst) wires syndrome bits from an embedded circuit into QUA variables in the **same** `with program():` block.

## Measurement outcomes vs what you stream to OPNIC

Hybrid QEC involves **two separate classical pipelines**:

| Pipeline | Where it lives | What it holds |
|----------|----------------|---------------|
| **Circuit measurement** | [`comp.outputs`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable.rst) on [`QuaCircuitCompilation`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.rst) | Raw discriminated bits from the embedded syndrome circuit — compiler-local, not OPNIC-emitted |
| **Host transport** | Runtime [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) / [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) | Whatever classical quantity **you choose** to assign and `stream_back()` |

Matching an OPNIC struct field name to a creg name is optional. Even when the names match, they refer to **different QUA variables**. The struct that streams back to OPNIC is **not required** — and often **should not** — be a 1:1 copy of the raw measurement register.

Typical QEC quantities streamed to the host include:

- **Detection events** — per-bit XOR between consecutive syndrome rounds (did this stabilizer flip?)
- **Packed syndrome integers** — compact encoding for a classical decoder lookup table
- **Aggregated counters or flags** — derived entirely in QUA from measurement bits

The pattern is: read raw outcomes from `comp.outputs`, **process in QUA**, then assign the **derived** result into a runtime `ParameterTable` and call `stream_back()`. See [Measurement outputs — QEC](measurement_outputs.md#qec--in-qua-processing-first) for the locality model.

## Why hybrid QEC is hard without this toolbox

Qiskit models a `QuantumCircuit` as a self-contained unit: gates, classical registers, and optional control flow all live **inside** the circuit boundary. QEC cycles do not fit that picture. Each round is really three phases:

1. **Syndrome measurement** — a circuit fragment on the OPX.
2. **Classical decoding** — often heavy (MWPM, lookup tables, …), running on a host or server, **outside** any circuit.
3. **Recovery** — a second circuit fragment whose parameters depend on the decoder output.

Qiskit has no first-class way to express that sandwich: two distinct circuit parts linked by real-time classical I/O. Unrolling every branch into one giant circuit, or gluing phases together with ad-hoc re-submission scripts, obscures the measure → decode → recover contract and scales poorly as cycle count or decoder latency grows.

QUA + [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) is aimed at exactly this gap — a stable boundary between circuit fragments and the classical processing that sits between them:

| Phase | Role of this toolbox |
|-------|------------------------|
| Syndrome measure | Embed a template Qiskit circuit in a QUA `for_` loop via [`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) |
| Stream to decoder | `stream_back()` detection events or packed syndromes through a runtime [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) / [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) |
| Classical decode | Host-side (as intended) — not forced into the circuit |
| Push recovery | `push_to_opx()` on recovery parameters, then `rcv()` in QUA |
| Recovery | Embed the recovery template with the updated [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) |

The same syndrome and recovery circuits are reused every cycle; only the streamed classical data and pushed recovery parameters change.

## ParameterTable roles in QEC

### Declare the contract once

```python
from qiskit.circuit.classical.expr import Var
from qiskit_qm_provider import Parameter, ParameterTable, Direction, InputType

num_cregs = len(syndrome_circuit.cregs)

syndrome_data = Parameter(
    "syndrome_data",
    [0] * num_cregs,  # one packed state_int slot per classical register
    input_type=InputType.INPUT_STREAM,  # or InputType.OPNIC
    direction=Direction.INCOMING,
)

recovery_vars = ParameterTable.from_qiskit(
    recovery_circuit,
    input_type=InputType.INPUT_STREAM,
    filter_function=lambda p: isinstance(p, Var),  # real-time Vars only, not compile-time Parameters
)
```

Inside QUA:

```python
recovery_vars.declare()
syndrome_data.declare(declare_stream=True)
```

### Stream syndrome out (same program, right after embedding)

[`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) returns a [`QuaCircuitCompilation`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.rst). Classical outcomes are exposed on **`comp.outputs`** ([`MeasurementOutcomeTable`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable.rst)) — compiler-local handles wired from `result_program`, distinct from the runtime `syndrome_data` parameter you stream to OPNIC.

```python
from qm.qua import program, assign

ancilla_creg = syndrome_circuit.cregs[0]

with program() as qec_prog:
    comp = backend.quantum_circuit_to_qua(syndrome_circuit)

    # Bridge: copy packed syndrome from compilation outputs into OPNIC transport
    assign(syndrome_data.var[0], comp.outputs.state_ints[ancilla_creg.name])
    syndrome_data.stream_back(reset=True)
```

For multiple classical registers, index `comp.outputs` per creg name (or use the bulk accessor):

```python
for i, creg in enumerate(syndrome_circuit.cregs):
    assign(syndrome_data.var[i], comp.outputs.state_ints[creg.name])
syndrome_data.stream_back(reset=True)
```

| Access on `comp.outputs` | Role in QEC |
|--------------------------|-------------|
| `comp.outputs["c"]` | Per-bit QUA bool array — raw stabilizer readout |
| `comp.outputs.state_ints["c"]` | Lazy-packed `int` (LSB = bit 0) — usual handle for decoding |
| `comp.outputs.get_parameter("c").size` | Number of syndrome bits in register `c` |
| `comp.outputs.streams["c"]` | Per-field stream for `save(...)` / host `stream_processing()` |

See [Measurement outputs guide](measurement_outputs.md) for the full accessor contract. Legacy [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst) returns the same handles via a dict shim (`meas[creg.name]["state_int"]` ≡ `comp.outputs.state_ints[creg.name]`).

This streams the **packed raw syndrome** each round. When the decoder needs **detection events** (bit flips between rounds) instead, see [Detection events — consecutive-round XOR](#detection-events--consecutive-round-xor) below.

### Detection events — consecutive-round XOR

Many decoders consume **detection events** (syndrome changes between rounds), not the raw stabilizer readout each round. Compute them in QUA and stream only the derived bits to OPNIC.

**Setup (Python, before `with program():`):** declare two runtime tables — one per classical register in the syndrome circuit:

```python
from qiskit_qm_provider import ParameterTable, InputType, Direction

def _bool_table(name: str, circuit, *, stream: bool = False):
    """One bool array field per creg, sized to match the circuit."""
    fields = {}
    for creg in circuit.cregs:
        spec: tuple = ([False] * creg.size, bool)
        if stream:
            spec = ([False] * creg.size, bool, InputType.INPUT_STREAM, Direction.INCOMING)
        fields[creg.name] = spec
    return ParameterTable(fields, name=name)

previous_measurement_outcomes = _bool_table("prev_meas", syndrome_circuit)
syndrome_data = _bool_table("syndrome_data", syndrome_circuit, stream=True)
```

- `previous_measurement_outcomes` — last round's raw bits (updated in-place each round).
- `syndrome_data` — **detection events** sent to the host via `stream_back()`.
- Current-round raw bits come directly from **`comp.outputs`** after each `quantum_circuit_to_qua` call — no separate staging table required.

**In-QUA update (inside `with program():`, after each syndrome measurement):**

```python
from qm.qua import declare, for_, assign

def update_syndrome_streams(
    circuit,
    comp,
    previous_measurement_outcomes: ParameterTable,
    syndrome_data: ParameterTable,
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

**Walkthrough:**

1. **Embed and measure** — `comp = backend.quantum_circuit_to_qua(syndrome_circuit)` runs the syndrome circuit; `comp.outputs[creg.name]` holds fresh bool outcomes for this round.
2. **XOR for detection events** — for each bit `j`, `syndrome_param[j] = prev_meas[j] ^ meas_reg[j]`. A `1` means the stabilizer outcome **changed** since the last round; `0` means no flip. This is the quantity many MWPM / lookup decoders expect.
3. **Advance history** — `prev_meas[j] = meas_reg[j]` so the next round compares against this round's raw readout from `comp.outputs`.
4. **Stream derived data only** — `syndrome_data.stream_back(reset=True)` pushes detection events to OPNIC, **not** the raw `comp.outputs` register. The host decoder never needs the compiler-local handles.

On the first round, initialize `previous_measurement_outcomes` to your experiment's baseline (often all zeros) before entering the QEC loop.

**Full loop sketch:**

```python
from qm.qua import program, declare, for_

with program() as qec_prog:
    previous_measurement_outcomes.declare()
    syndrome_data.declare(declare_stream=True)
    round = declare(int)

    with for_(round, 0, round < num_cycles, round + 1):
        comp = backend.quantum_circuit_to_qua(syndrome_circuit)
        update_syndrome_streams(
            syndrome_circuit,
            comp,
            previous_measurement_outcomes,
            syndrome_data,
        )
        # host decodes detection events → push recovery_vars ...
```

### Packed `state_int` for large registers

When a classical register has many bits, streaming or retaining the full bool chain is expensive on FPGA memory and host buffers. Prefer the lazy-packed **`state_int`** scalar from **`comp.outputs.state_ints`** — one `int` per creg, **LSB = bit index 0** (Qiskit convention). See [Measurement outputs — accessor contract](measurement_outputs.md#accessor-contract-aligned-with-parametertable).

Use `state_int` when you need a **compact label** for the outcome, not individual bit logic:

- Stream one integer per round to OPNIC instead of `creg.size` bool streams.
- Index a QUA histogram or lookup table of size `2**creg.size` on device.
- Host-side `stream_processing()` with a 1-D buffer of length `2**creg.size` instead of a `(n_rounds, creg.size)` bool array.

**Example — stream packed syndromes and accumulate histograms in QUA:**

```python
from qm.qua import program, declare, for_, assign, stream_processing

# Python: one int field per creg for OPNIC transport
syndrome_int = Parameter(
    "syndrome_int",
    0,
    input_type=InputType.INPUT_STREAM,
    direction=Direction.INCOMING,
)

# Python: per-creg histogram on device (size 2**n_bits)
hist_table = ParameterTable({
    creg.name: ([0] * (2 ** creg.size), int)
    for creg in syndrome_circuit.cregs
}, name="syndrome_hist")

ancilla = syndrome_circuit.cregs[0].name

with program() as qec_prog:
    syndrome_int.declare(declare_stream=True)
    hist_table.declare()
    round = declare(int)

    with for_(round, 0, round < num_cycles, round + 1):
        comp = backend.quantum_circuit_to_qua(syndrome_circuit)

        for creg in syndrome_circuit.cregs:
            state_int = comp.outputs.state_ints[creg.name]
            # Compact integer label — avoids retaining creg.size bool vars for streaming
            assign(hist_table[creg.name][state_int], hist_table[creg.name][state_int] + 1)

        assign(syndrome_int.var, comp.outputs.state_ints[ancilla])
        syndrome_int.stream_back(reset=True)

    with stream_processing():
        for creg in syndrome_circuit.cregs:
            target_dim = 2 ** creg.size
            hist_table.get_parameter(creg.name).stream_processing(buffer=(target_dim,))
        syndrome_int.stream_processing()
```

**When to use which representation:**

| Representation | Best for |
|----------------|----------|
| Per-bit bool (`comp.outputs["c"]`, `ParameterTable` of bool arrays) | In-QUA bit logic — XOR detection events, parity checks, conditional feedback on individual stabilizers |
| Packed `state_int` | Streaming to host, histogramming, decoder lookup tables, any workflow where the full bit string is treated as one label |

You can combine both: use bool arrays (or XOR) for real-time feedback inside the QUA loop, and assign `state_int` into an OPNIC `Parameter` when the host only needs the compact syndrome label.

### Push recovery parameters in

```python
recovery_vars.push_to_opx(param_dict, job, verbosity=0)
# Prefer JobApi for IO; `qm` is only needed for older job objects.
```

In QUA:

```python
recovery_vars.rcv()
backend.quantum_circuit_to_qua(recovery_circuit, recovery_vars)
```

## End-to-end loop

1. Author syndrome and recovery circuits in Qiskit.
2. Declare [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) / [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) for the **derived** classical quantities you will stream (detection events, packed integers, etc.) — not necessarily raw measurement registers.
3. Inside `with program():`, call [`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) for the syndrome circuit and read outcomes from **`comp.outputs`**.
4. Process outcomes in QUA (XOR for detection events, or `comp.outputs.state_ints[...]` for packed syndromes), then `stream_back()` from the runtime table.
5. On the host, decode and `push_to_opx()` recovery parameters.
6. `rcv()` on the recovery table and embed the recovery circuit.

The full QEC program skeleton lives in the [repository README](https://github.com/arthurostrauss/qiskit-qm-provider#error-correction-and-parameter-table).

## Related

- **Guides:** [Measurement outputs (`comp.outputs`)](measurement_outputs.md), [Backend — embedding](backend.md#embed-in-qua-hybrid), [Parameter Table](parameter_table.md), [Workflows — hybrid](workflows.md#hybrid-qua-qiskit-programs-embedding-circuits-in-qua)
- **API:** [QuaCircuitCompilation](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.rst), [Parameter Table reference](apidocs/qm_parameter_table.rst), [Backend reference](apidocs/qm_backend.rst)
- **Example:** [Error correction in README](https://github.com/arthurostrauss/qiskit-qm-provider#error-correction-and-parameter-table)
