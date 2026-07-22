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

- `push_to_opx(param_dict, job=..., qm=...)` — send values to the OPX (`qm` is optional legacy; prefer `JobApi` for IO).
- `fetch_from_opx(job, ...)` — retrieve streamed results.

## Constructing a `ParameterTable`

We recommend building explicit [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) objects first and passing a **list** to `ParameterTable`. That keeps `input_type`, `direction`, and `qua_type` visible at each field and matches how OPNIC / Quarc emission inspects the table.

You can also pass a **dictionary** when a quick inline spec is enough. Every key is the parameter name; each value follows one of the shapes below.

### Dictionary value shapes

| Value | Meaning |
|---|---|
| `scalar` or `[v0, v1, …]` | Initial value only; `qua_type` is inferred from the value (`int` → `int`, `float` → `fixed`, list of floats → `fixed` array, etc.). |
| `(value, qua_type)` | Explicit QUA type: `int`, `fixed`, `bool`, or the string `"int"` / `"fixed"` / `"bool"`. |
| `(value, qua_type, input_type)` | Adds how the host talks to the OPX: `InputType.INPUT_STREAM`, `InputType.IO1`, `InputType.IO2`, or `InputType.OPNIC`. |
| `(value, qua_type, input_type, direction)` | Required fourth field when `input_type` is `InputType.OPNIC` (`Direction.INCOMING`, `OUTGOING`, or `BOTH`). |

**Rules:**

- Every parameter in one table must share the same `input_type` (and the same `direction` for OPNIC).
- A list or 1D numpy array as `value` defines an array parameter; length is `len(value)`.
- OPNIC tables cannot use `add_parameters` / `remove_parameter` after the struct has been emitted (`declare()` in Flow A, or a pre-bound handle in Flow B).

### Recommended — list of `Parameter` objects

```python
from qm.qua import fixed
from qiskit_qm_provider import Parameter, ParameterTable, Direction, InputType

mu = Parameter("mu", [0.0, 0.0], qua_type=fixed, input_type=InputType.OPNIC, direction=Direction.INCOMING)
sigma = Parameter("sigma", [0.1, 0.1], qua_type=fixed, input_type=InputType.OPNIC, direction=Direction.INCOMING)

policy_params = ParameterTable([mu, sigma], name="PolicyParams")
```

Compile-time angles (no streaming — assigned in QUA or bound from calibration):

```python
gate_params = ParameterTable(
    [
        Parameter("theta", 0.5, qua_type=fixed),
        Parameter("phi", 0.0, qua_type=fixed),
    ],
    name="gate_params",
)
```

### Dictionary shorthand — scalar compile-time

```python
# qua_type inferred: 0.0 → fixed, 1 → int, False → bool
cal_table = ParameterTable(
    {
        "theta": 0.5,
        "n_reps": 100,
        "flag": False,
    },
    name="cal",
)
```

### Dictionary shorthand — explicit type

```python
from qm.qua import fixed

cal_table = ParameterTable(
    {
        "theta": (0.5, fixed),
        "n_reps": (100, int),
        "flag": (False, bool),
    },
    name="cal",
)
```

### Dictionary shorthand — input stream / IO

```python
# Host pushes values before each rcv() in QUA
stream_table = ParameterTable(
    {
        "input_state_0": (0, int, InputType.INPUT_STREAM),
        "observable_0": (0, int, InputType.INPUT_STREAM),
    },
    name="input_state_vars",
)

# IO registers (job must be paused; see Parameter.push_to_opx)
io_table = ParameterTable(
    {
        "sync_flag": (False, bool, InputType.IO1),
        "status": (0, int, InputType.IO2),
    },
    name="io_flags",
)
```

String literals work the same as enums: `"INPUT_STREAM"`, `"IO1"`, `"IO2"`.

### Dictionary shorthand — OPNIC packet (four-tuple)

```python
policy_params = ParameterTable(
    {
        "mu": ([0.0, 0.0], "fixed", "OPNIC", "INCOMING"),
        "sigma": ([0.1, 0.1], "fixed", "OPNIC", "INCOMING"),
    },
    name="PolicyParams",
)
```

### Growing a table before `declare()` — `add_parameters`

Build the core fields first, then attach more `Parameter` instances **before** the table is emitted (before `declare()` for OPNIC, or before any Flow-B handle is bound):

```python
base = ParameterTable(
    [Parameter("mu", [0.0, 0.0], input_type=InputType.OPNIC, direction=Direction.INCOMING)],
    name="PolicyParams",
)

base.add_parameters(
    Parameter("sigma", [0.1, 0.1], input_type=InputType.OPNIC, direction=Direction.INCOMING)
)

# Or merge another table's fields:
extra = ParameterTable(
    [Parameter("bias", 0.0, input_type=InputType.OPNIC, direction=Direction.INCOMING)],
    name="bias_only",
)
base.add_table(extra)  # adds Parameter("bias", ...)
```

After `declare()` (or once a Quarc struct handle is bound), `add_parameters` and `remove_parameter` raise — Quarc struct layout is append-only.

### Other constructors

- [`ParameterTable.from_qiskit(qc, ...)`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) — split circuit `Parameter` / `Var` inputs into tables (see tomography example below).
- [`ParameterTable.from_spec(spec_dict)`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) — round-trip from serialized module state.

## Minimal example

```python
from qiskit_qm_provider import Parameter, ParameterTable, Direction, InputType
```

Represents a single parameter. Shares the same canonical interface as `ParameterTable` (see table above).

```python
# Streamed syndrome integer (host ← device)
syndrome_data = Parameter("syndrome_data", 0, input_type=InputType.INPUT_STREAM)
```

**Parameter `__init__` arguments:**
- `name`: Parameter name.
- `value`: Initial Python-side value (also used by `push_to_opx()` when called with no argument).
- `qua_type`: `int`, `float`, `bool`, or `fixed`.
- `input_type`: `InputType` enum.
- `direction`: `Direction` enum (for OPNIC).

```python
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

## Splitting one circuit into multiple tables (`filter_function`)

Quantum process tomography and direct fidelity estimation repeat the same pulse program many times while varying two independent choices: which **input state** to prepare (e.g. the six Pauli eigenstates on the Bloch sphere) and which **measurement observable** to read out (e.g. Pauli X, Y, Z). A compact real-time circuit encodes both choices as `switch` blocks driven by classical input variables, but the host must not load every index at once — the observable can stay fixed while input states are swept, and only then move to the next observable.

[`ParameterTable.from_qiskit()`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) walks **both** symbolic [`Parameter`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.Parameter) objects (gate angles under calibration) and real-time [`Var`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.classical.expr.Var) inputs (via `qc.add_input(...)`). The optional **`filter_function`** selects which objects belong in a given table:

```python
filter_function: Callable[[Parameter | Var], bool] | None
```

Call `from_qiskit` several times on the **same** circuit with different filters to obtain disjoint tables that share one compiled circuit but are **`rcv()`'d at different depths** in the QUA control flow. The same filter can be passed to [`ParameterTable.rcv()`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) to load only a subset of an already-declared table (not supported for OPNIC tables).

Typical split for tomography:

| Table | Filter | Loaded when |
|---|---|---|
| `observable_vars` | `"observable" in x.name` | Start of each observable group (outer loop) |
| `input_state_vars` | `"input" in x.name` | Start of each input-state group (inner loop) |
| `gate_params` | `isinstance(x, Parameter)` | Assigned in QUA before the shot loop (calibration angles) |

The host streams one `(observable_0, input_state_0)` pair per inner iteration, sweeping all **3 × 6 = 18** combinations — without storing the full index grid on the FPGA.

### Single-qubit process tomography (Qiskit side)

The circuit below is a minimal tomography cell for one qubit: a **Pauli-6 input-state** switch, a parametrized gate under test (`x_cal`), a **3-way measurement-basis** switch, then measure and reset. Real-time indices enter through `add_input`; the gate angle is a single symbolic `Parameter` named `a`.

```python
from qiskit.circuit import QuantumCircuit, Gate, ClassicalRegister, Parameter as QiskitParameter
from qiskit.circuit.classical import types
from qiskit_qm_provider import Parameter, ParameterTable, InputType

# --- Real-time circuit (one qubit) -----------------------------------------
a = QiskitParameter("a")
x_cal = Gate("x_cal", 1, [a])

qc = QuantumCircuit(1, name="single_qubit_tomography")
meas = ClassicalRegister(1, name="meas_target_0")
qc.add_register(meas)

input_state_0 = qc.add_input("input_state_0", types.Uint(32))
observable_0 = qc.add_input("observable_0", types.Uint(32))

# Pauli-6 input-state preparation (indices 0 … 5)
with qc.switch(input_state_0) as case_input:
    with case_input(0):
        qc.delay(16, 0)          # |0⟩
    with case_input(1):
        qc.x(0)                    # |1⟩
    with case_input(2):
        qc.h(0)                    # |+⟩
    with case_input(3):
        qc.h(0)
        qc.z(0)                    # |−⟩
    with case_input(4):
        qc.h(0)
        qc.s(0)                    # |+i⟩
    with case_input(5):
        qc.h(0)
        qc.sdg(0)                  # |−i⟩

qc.append(x_cal, [0])

# Pauli measurement basis (0: Z, 1: X, 2: Y)
with qc.switch(observable_0) as case_obs:
    with case_obs(0):
        qc.delay(16, 0)
    with case_obs(1):
        qc.h(0)
    with case_obs(2):
        qc.sdg(0)
        qc.h(0)

qc.measure(0, meas[0])
qc.reset(0)

# --- Split one circuit into role-specific tables ---------------------------
input_state_vars = ParameterTable.from_qiskit(
    qc,
    input_type=InputType.INPUT_STREAM,
    filter_function=lambda x: "input" in x.name,
    name="input_state_vars",
)
observable_vars = ParameterTable.from_qiskit(
    qc,
    input_type=InputType.INPUT_STREAM,
    filter_function=lambda x: "observable" in x.name,
    name="observable_vars",
)
gate_params = ParameterTable.from_qiskit(
    qc,
    input_type=None,  # assigned in QUA (calibration angle)
    filter_function=lambda x: isinstance(x, QiskitParameter),
    name="gate_params",
)

n_shots = Parameter("n_shots", 100, input_type=InputType.INPUT_STREAM)
```

On the Python host, `push_to_opx` is called in lockstep with the QUA loops: for each of the 3 observables and 6 input states, push `observable_0`, `input_state_0`, and `n_shots`. Histograms from each pair are combined offline into expectation values or a process matrix.

### Matching QUA program (nested loops, staged loading)

The template below exhausts all **3 × 6** tomography settings while keeping only **two** integer control-flow variables live (`input_state_0`, `observable_0`) plus one gate angle — not 18 copies of every index stored on the FPGA at once.

```python
from qm.qua import program, declare, for_, fixed
from qiskit_qm_provider import QMBackend

backend = ...  # QMBackend with the transpiled qc above
calibration_angle = declare(fixed)  # set once, or loaded from the host before the sweep

with program() as tomography_prog:
    input_state_vars.declare()
    observable_vars.declare()
    gate_params.declare()
    n_shots.declare()

    o_idx = declare(int)
    i_idx = declare(int)
    shots = declare(int)

    backend.init_macro()

    with for_(o_idx, 0, o_idx < 3, o_idx + 1):
        observable_vars.rcv()      # host pushes observable_0 ∈ {0, 1, 2}

        with for_(i_idx, 0, i_idx < 6, i_idx + 1):
            input_state_vars.rcv()  # host pushes input_state_0 ∈ {0, …, 5}
            n_shots.rcv()
            gate_params.get_parameter("a").assign(calibration_angle)

            with for_(shots, 0, shots < n_shots.var, shots + 1):
                backend.quantum_circuit_to_qua(qc, circuit_variables)
                # accumulate counts into histogram for this (observable, input_state) pair …
```

**Why the filters matter:** a single unfiltered `ParameterTable.from_qiskit(qc)` would merge `input_state_0`, `observable_0`, and `a`. One `rcv()` would advance every input stream at once, breaking the nested-loop schedule. Splitting by `filter_function` keeps each stream aligned with the control-flow depth where its Qiskit `switch` is evaluated — the natural structure for tomography experiments on long-running QUA programs.

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
- **[`QUA2DArray`](apidocs/stubs/qiskit_qm_provider.parameter_table.QUA2DArray.rst) / [`QUAArray`](apidocs/stubs/qiskit_qm_provider.parameter_table.QUAArray.rst)** — multi-index parameter memory (flattened QUA arrays).
- **`ParameterVector` note:** OpenQASM 3 exports `ParameterVector` elements as individual parameters; `from_qiskit` handles this transparently.

## Related

- **Guide:** [Measurement outputs](measurement_outputs.md), [Backend — embedding](backend.md#embed-in-qua-hybrid), [Error correction](error_correction.md)
- **API:** [Parameter Table reference](apidocs/qm_parameter_table.rst), [Backend — measurement outputs](apidocs/qm_backend.rst#circuit-compilation-and-measurement-outputs)
- **Workflows:** [Hybrid embedding](workflows.md#4-hybrid-quaqiskit-programs-embedding-circuits-in-qua)
