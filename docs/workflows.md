---
title: Workflows and Examples
nav_order: 2
parent: Home
---

# Workflows and Examples

This page walks through the **main workflows** supported by `qiskit-qm-provider` and points you to
the concrete examples in the repository. The goal is to show how the toolbox helps you **stay in
Qiskit** while taking advantage of **QOP/QUA** for execution and real‑time control.

## 1. Running Qiskit circuits on QM hardware or simulators

### 1.1 Local hardware with `QMProvider`

Use `QMProvider` when you have a **local QOP stack** (OPX/OPX+ hardware) and a **QuAM state**
stored on disk.

Typical flow:

1. Create a `QMProvider` pointing to your QuAM state folder.
2. Optionally provide your own `QuamRoot` subclass (`quam_cls`) and `QMBackend` subclass (`backend_cls`).
3. Use the backend with standard Qiskit workflows: `backend.run()`, `QMSamplerV2`, `QMEstimatorV2`.

See:

- API docs: [Providers](providers.md) (section on `QMProvider`).
- API docs: [Backend & Utilities](backend.md).
- Examples:
  - `examples/sampler_workflow.py`
  - `examples/estimator_workflow.py`

### 1.2 QM SaaS simulator with `QmSaasProvider`

Use `QmSaasProvider` when you want to run on a **cloud simulator** managed by Quantum Machines.
You get the same backend interface, but the QOP instance runs in the cloud.

Typical flow:

1. Install the SaaS extra: `pip install qiskit-qm-provider[qm_saas]`.
2. Create `QmSaasProvider(email=..., password=..., host=...)`.
3. Load a QuAM machine with `get_machine()` or directly call `get_backend()`.
4. Use `backend.run()` or the primitives (`QMSamplerV2`, `QMEstimatorV2`) as usual.

See:

- API docs: [Providers](providers.md) (section on `QmSaasProvider`).
- Examples:
  - `examples/sampler_workflow.py` (can be adapted to SaaS by switching the provider).

### 1.3 IQCC devices with `IQCCProvider`

Use `IQCCProvider` when you want to run on **IQCC hardware** via `iqcc-cloud-client`.
IQCC backends are flux‑tunable transmon machines and are exposed as `FluxTunableTransmonBackend`
instances.

Typical flow:

1. Install the IQCC extra: `pip install qiskit-qm-provider[iqcc]`.
2. Create `IQCCProvider(api_token=...)`.
3. Optionally fetch a QuAM machine explicitly with:

   ```python
   machine = provider.get_machine(
       "arbel",
       quam_state_folder_path="/path/to/quam/state",  # or rely on QUAM_STATE_PATH
       # quam_cls=CustomIQCCQuam,  # optional: inject a specific Quam class
   )
   ```

4. Obtain a backend either from a device name or from a pre-loaded machine:

   ```python
   # From a device name (loads QuAM under the hood)
   backend = provider.get_backend(
       "arbel",
       quam_state_folder_path="/path/to/quam/state",  # optional, falls back to QUAM_STATE_PATH
       # quam_cls=CustomIQCCQuam,
   )

   # Or from an already-loaded machine
   backend = provider.get_backend(machine)
   ```

5. Use Qiskit Experiments or custom circuits against this backend.

See:

- API docs: [Providers](providers.md) (section on `IQCCProvider`).
- Example:
  - `examples/iqcc_t1_experiment.py` – T1 measurement using Qiskit Experiments + IQCC.

## 2. Calibrations and custom gates

### 2.1 Calibrations and pulse‑level workflows

For Qiskit 1.x environments with Pulse enabled, you can:

- Use `FluxTunableTransmonBackend` to expose qubit and coupler channels as Qiskit Pulse channels.
- Convert pulse schedules to QUA via `schedule_to_qua_macro`.
- Use `add_basic_macros` to populate QuAM with standard operations (`x`, `sx`, `cz`, `measure`, etc.).

See:

- API docs: [Backend & Utilities](backend.md) (sections on `schedule_to_qua_macro` and `add_basic_macros`).
- Example:
  - `examples/circuit_calibrations_pulse.py`.

### 2.2 Custom gates via `QMInstructionProperties`

In Qiskit 2.x (without Pulse), the recommended way to express **custom calibrated operations** is
through `QMInstructionProperties`:

1. Define a new gate at the Qiskit circuit level (e.g. a parametric two‑qubit gate).
2. Write a QUA macro that implements the calibrated behavior.
3. Attach the QUA macro and timing/error information via `QMInstructionProperties`.
4. Update the backend target with `backend.target.add_instruction(...)` and call `backend.update_target()`.

This keeps the **Qiskit Target** and the **qm_qasm compiler mapping** in sync, so both
`backend.run()` and `backend.quantum_circuit_to_qua()` understand your new gate.

See:

- API docs: [Backend & Utilities](backend.md) (section on `QMInstructionProperties`).
- Example snippet: in the main README and on the home page under “Custom calibrations”.

## 3. Primitives: Sampler and Estimator on QOP

`QMEstimatorV2` and `QMSamplerV2` implement Qiskit’s V2 primitives on top of QM backends.
They are designed to:

- re‑use QuAM‑derived Targets for transpilation,
- stream parameters in real time via `InputType` (input streams, IO, DGX_Q),
- and map shot budgets to QOP shots and QUA loops.

### 3.1 Generated QUA programs (and how to inspect them)

`QMSamplerV2`, `QMEstimatorV2`, and the `backend.run()` interface are meant to let you stay in
**Qiskit-land** while the provider automatically generates the **QUA program** required to execute
your circuits on QOP.

If you need to debug what is actually being sent to QOP, you can inspect the generated QUA program
from the returned job object:

- For primitives: `job = sampler.run(...)` or `job = estimator.run(...)`
- For the backend: `job = backend.run(...)`

In all cases, the generated QUA `Program` is available as `job.program`. You can pretty-print it as
a QUA script using `qm.generate_qua_script`:

```python
from qm import generate_qua_script

print(generate_qua_script(job.program))
```

Below is a complete snippet showing the workflow end-to-end for `QMSamplerV2`, `QMEstimatorV2`, and
`backend.run()`, including how to print the auto-generated QUA program for each job:

```python
from qm import generate_qua_script

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from qiskit_qm_provider import QMProvider, QMSamplerV2, QMSamplerOptions, QMEstimatorV2, QMEstimatorOptions

# 1) Get a backend from a provider (local QuAM folder example)
provider = QMProvider(state_folder_path="/path/to/quam/state")
backend = provider.get_backend()

# 2) Define a small circuit
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
qc = transpile(qc, backend)

# --- Sampler primitive ---
sampler = QMSamplerV2(backend=backend, options=QMSamplerOptions(default_shots=256))
sampler_job = sampler.run([qc])

print("=== Sampler: generated QUA program ===")
print(generate_qua_script(sampler_job.program))

sampler_result = sampler_job.result()
print("Sampler result:", sampler_result)

# --- Estimator primitive ---
# (Use an observable compatible with the circuit's number of qubits.)
obs = SparsePauliOp.from_list([("Z", 1.0)])
estimator = QMEstimatorV2(backend=backend, options=QMEstimatorOptions())
estimator_job = estimator.run([(qc.remove_final_measurements(inplace=False), obs, [])])

print("=== Estimator: generated QUA program ===")
print(generate_qua_script(estimator_job.program))

estimator_result = estimator_job.result()
print("Estimator result:", estimator_result)

# --- Backend.run() (traditional Qiskit backend interface) ---
backend_job = backend.run(qc, shots=256)

print("=== backend.run(): generated QUA program ===")
print(generate_qua_script(backend_job.program))

backend_result = backend_job.result()
print("backend.run() result:", backend_result)
```

Typical flow:

1. Build or obtain a backend (`QMProvider`, `QmSaasProvider`, or `IQCCProvider`).
2. Create options (`QMEstimatorOptions` / `QMSamplerOptions`) choosing the appropriate `InputType`.
3. Run the primitive on circuit/observable pairs.

See:

- API docs: [Primitives](primitives.md).
- Examples:
  - `examples/sampler_workflow.py`
  - `examples/estimator_workflow.py`

## 4. Hybrid QUA/Qiskit programs (embedding circuits in QUA)

One of the key workflows is to treat Qiskit circuits as **building blocks** inside larger QUA
programs:

1. Define and transpile a `QuantumCircuit` for your backend.
2. Use `backend.quantum_circuit_to_qua(qc, param_table=...)` inside a QUA `program()` context.
3. Use `get_measurement_outcomes` to recover classical results as QUA variables and streams.
4. Combine this with QOP control flow (loops, conditionals) and streaming (input/output).

This is particularly powerful for:

- **error correction cycles** (syndrome measurement + recovery),
- **closed‑loop calibration** and optimal control,
- **DGX Quantum** or other hybrid classical‑quantum control loops.

See:

- API docs:
  - [Backend & Utilities](backend.md) (sections on `quantum_circuit_to_qua` and `get_measurement_outcomes`).
  - [Parameter Table](parameter_table.md).
- Examples:
  - Error‑correction example in the README and the dedicated
    [Error‑Correction Workflow](error_correction.md) page.

## 5. Error‑correction workflow (overview)

The error‑correction use case is where hybrid workflows really shine:

- You repeatedly:
  - prepare an encoded state,
  - run a **syndrome‑measurement circuit** (authored in Qiskit),
  - extract a classical syndrome,
  - select or compute a **recovery** circuit or set of parameters,
  - apply the recovery,
  - and stream out data for later analysis.

The usual pain points are:

- wiring all the classical data between repeated QUA loops and Python,
- avoiding a combinatorial explosion of Qiskit circuits just to represent classical branches,
- keeping parameter names and data layouts consistent between Qiskit and QUA.

The provider’s **ParameterTable** and helpers (`get_measurement_outcomes`, QUA‑side control flow)
are designed to address exactly these issues.

For a step‑by‑step explanation of this pattern, see the dedicated
[Error‑Correction Workflow](error_correction.md) page, which builds on the example in the README
and explains:

- the typical challenges in hybrid error‑correction programs, and  
- how `ParameterTable` makes the classical‑quantum boundary explicit and manageable.

