---
title: Error‑Correction Workflow
nav_order: 7
parent: Home
---

# Error‑Correction Workflow with ParameterTable

Error‑correction experiments are a **stress‑test for hybrid programming**:

- you repeat a **syndrome‑measurement circuit** many times,
- extract classical information (the syndrome),
- use that information to choose or parametrize a **recovery**,
- and log data for later analysis — all while keeping the quantum hardware busy.

This page explains:

1. **What usually makes hybrid error‑correction workflows painful**, and  
2. **How the `ParameterTable` interface helps** by making the classical–quantum boundary explicit.

It builds on the example shown in the README and in the Home page.

## 1. The usual challenges in hybrid error‑correction programs

### 1.1 Two worlds with different assumptions

When writing an error‑correction loop, you typically have:

- **Qiskit circuits** describing:
  - state preparation / encoding,
  - syndrome‑measurement steps,
  - and (optionally) recovery circuits.
- **QUA programs** (or other control code) that:
  - manage loops over shots and cycles,
  - talk to the hardware (pulses, measurements, streaming),
  - and push / pull data to and from the classical host.

These two worlds use different abstractions:

- Qiskit assumes a **single circuit run** at a time, with parameters bound
  either at compile time or at primitive submission time;
- QUA assumes **long‑running programs** with nested loops, streaming, and explicit buffers.

Bridging them manually often leads to a lot of glue code.

### 1.2 Classical data flow is awkward to express

In a realistic error‑correction experiment you might need to:

- record a **syndrome bitstring** (or integer) every cycle,
- pass that syndrome to a **classical controller** (on DGX, another server, or Python),
- receive back **recovery parameters** or decisions,
- and then update QUA‑side variables accordingly.

With “plain” Qiskit circuits alone, you quickly run into:

- **circuit explosion**: one circuit per possible branch or per time step;
- or **ad‑hoc data plumbing**: many custom classical registers, custom JSON blobs, and
  brittle indexing on both sides.

### 1.3 Keeping qubit‑level structure and parameters in sync

Error‑correction codes also care deeply about **which qubit** carries which role (data vs ancilla),
and **which parameters** (angles, thresholds, syndrome history) are used where.

Without a consistent interface, it is easy to end up with:

- parameter names that differ between Qiskit and QUA,
- mismatched dimensions (e.g. you think you streamed `num_cycles × num_qubits` values but the
  program expects a different shape),
- confusion about which piece of classical data corresponds to which Qiskit parameter.

## 2. How `ParameterTable` helps

`ParameterTable` is designed as a **single source of truth** for real‑time parameters and
classical data flowing between Qiskit and QUA. In an error‑correction workflow, it plays three
key roles:

1. **Declaring QUA‑side variables** with a clear name, type, and direction.
2. **Describing how classical data is streamed in and out** of the program.
3. **Providing a Python‑side handle** to push values to the OPX or fetch results.

### 2.1 Making the classical–quantum boundary explicit

Instead of scattering variables between ad‑hoc QUA declarations and Python dictionaries,
you define them centrally as `Parameter` objects (or in a dictionary) and bundle them into a
`ParameterTable`. For example:

```python
from qiskit_qm_provider import Parameter, ParameterTable, Direction, InputType

syndrome_data = Parameter(
    "syndrome_data",
    0,
    input_type=InputType.INPUT_STREAM,
    direction=Direction.INCOMING,
)

recovery_vars = ParameterTable.from_qiskit(
    recovery_circuit,
    input_type=InputType.INPUT_STREAM,
)
```

- `syndrome_data` represents the **syndrome integer** that will be sent back to the host.
- `recovery_vars` is a table of all parameters used by the **recovery circuit**.

QUA‑side, you simply call:

```python
recovery_vars.declare_variables()
syndrome_data.declare_variable()
syndrome_data.declare_stream()
```

to make these variables live inside the program.

### 2.2 Streaming syndrome data out

Inside the QUA loop, after running the **syndrome‑measurement circuit** via
`backend.quantum_circuit_to_qua`, you use `get_measurement_outcomes` to obtain a
`state_int` representing the measured syndrome:

```python
from qiskit_qm_provider.backend.backend_utils import get_measurement_outcomes

syndrome_meas_result = backend.quantum_circuit_to_qua(syndrome_circuit)
syndrome_meas_dict = get_measurement_outcomes(syndrome_circuit, syndrome_meas_result)
state_int_val = syndrome_meas_dict[ancilla_creg.name]["state_int"]

syndrome_data.assign(state_int_val)
syndrome_data.stream_back(reset=True)
```

Conceptually:

- `state_int_val` is the **hybrid handshake**: a compact representation of the measured syndrome.
- `syndrome_data.assign(...)` makes it available as a QUA variable.
- `syndrome_data.stream_back(...)` pushes it to the output stream so the host can read it.

On the Python side, you can then **fetch the stream** and apply any classical processing you
need (decoding, look‑up tables, machine‑learning models, etc.).

### 2.3 Streaming recovery parameters in

Once the host has computed new recovery parameters, you can push them back to the OPX using the
same `ParameterTable`:

```python
# Python side (after some classical processing of syndrome history)
param_dict = {
    "theta_0": value_for_cycle_0,
    "theta_1": value_for_cycle_1,
    # ...
}

recovery_vars.push_to_opx(param_dict, job, qm, verbosity=0)
```

In the QUA program, you then “wake up” these parameters at the right time:

```python
recovery_vars.load_input_values()
backend.quantum_circuit_to_qua(recovery_circuit, recovery_vars)
```

- `load_input_values()` reads the streamed values into the QUA variables.
- `quantum_circuit_to_qua(..., recovery_vars)` binds them into the recovery circuit.

Because the same `ParameterTable` definition is used on both sides, you avoid:

- name mismatches,
- shape mismatches,
- and ad‑hoc JSON or array indexing conventions.

### 2.4 Keeping the workflow scalable

By decoupling:

- **what** data you exchange (captured in `ParameterTable` and `Parameter` definitions), from  
- **how often / when** you exchange it (captured in QUA loops and host‑side logic),

the system stays scalable as you:

- increase the number of cycles,
- change the code distance or number of ancillas,
- or extend the set of recovery parameters.

You are not forced to re‑encode this structure in dozens of Qiskit circuits or ad‑hoc buffers:
the mapping remains localized in the parameter definitions.

## 3. Putting it all together

The full error‑correction example in the README (and referenced on the Home page) illustrates
this pattern:

1. A **syndrome‑measurement circuit** and a **recovery circuit** are authored in Qiskit.
2. `ParameterTable` and `Parameter` objects declare which classical values will flow between
   host and device.
3. `backend.quantum_circuit_to_qua` embeds the circuits inside a structured QUA program with
   loops over memory experiments and cycles.
4. `get_measurement_outcomes` and `syndrome_data.stream_back(...)` expose the syndrome data as
   a compact, streamable integer.
5. `recovery_vars.push_to_opx(...)` and `recovery_vars.load_input_values()` realize the
   **feedback step** by feeding processed parameters back into QUA.

The end result is a **hybrid error‑correction loop** where:

- Qiskit remains the language for circuits and gates,
- QUA remains the language for real‑time control and streaming,
- and `ParameterTable` is the **contract** that keeps both views aligned.

This is the kind of workflow `qiskit-qm-provider` is designed to make natural rather than painful.

