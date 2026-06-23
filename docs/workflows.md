# Workflows and Examples

This page is the **routing guide** for the main paths through `qiskit-qm-provider`. Each section points to a deeper guide, API reference, and runnable examples.

## 1. Running Qiskit circuits on QM hardware or simulators

Get a backend from a provider, transpile circuits, then use [`QMBackend.run()`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) or V2 primitives. Same Qiskit ergonomics; QOP executes the generated QUA underneath.

### 1.1 Local hardware with QMProvider

1. Create [`QMProvider`](apidocs/stubs/qiskit_qm_provider.providers.QMProvider.rst) with your QuAM state folder.
2. Optionally pass custom `quam_cls` / `backend_cls`.
3. Transpile and run.

- **Guide:** [Providers — QMProvider](providers.md#qmprovider-local-qop)
- **API:** [Providers reference](apidocs/qm_providers.rst)
- **Examples:** `examples/sampler_workflow.py`, `examples/estimator_workflow.py`

### 1.2 QM SaaS simulator with QmSaasProvider

1. `pip install qiskit-qm-provider[qm_saas]`
2. Create [`QmSaasProvider`](apidocs/stubs/qiskit_qm_provider.providers.qm_saas_provider.QmSaasProvider.rst).
3. Call `get_backend()` and run as usual.

- **Guide:** [Providers — QmSaasProvider](providers.md#qmsaasprovider-cloud-simulation)
- **Examples:** adapt `examples/sampler_workflow.py`

### 1.3 IQCC devices with IQCCProvider

1. `pip install qiskit-qm-provider[iqcc]`
2. Create [`IQCCProvider`](apidocs/stubs/qiskit_qm_provider.providers.iqcc_cloud_provider.IQCCProvider.rst).
3. Obtain a [`FluxTunableTransmonBackend`](apidocs/stubs/qiskit_qm_provider.backend.FluxTunableTransmonBackend.rst):

```python
machine = provider.get_machine(
    "arbel",
    quam_state_folder_path="/path/to/quam/state",  # or QUAM_STATE_PATH
)
backend = provider.get_backend(
    "arbel",
    quam_state_folder_path="/path/to/quam/state",
)
# or: backend = provider.get_backend(machine)
```

- **Guide:** [Providers — IQCCProvider](providers.md#iqccprovider-iqcc-cloud-devices)
- **Example:** `examples/iqcc_t1_experiment.py`

## 2. Calibrations and custom gates

### 2.1 Pulse-level workflows (Qiskit 1.x legacy)

When Pulse is available:

- Use [`FluxTunableTransmonBackend`](apidocs/stubs/qiskit_qm_provider.backend.FluxTunableTransmonBackend.rst) for QuAM ↔ Pulse channel mapping.
- Convert **gate pulse schedules** via [`schedule_to_qua_macro`](apidocs/stubs/qiskit_qm_provider.pulse.schedule_to_qua_macro.rst).
- Seed macros with [`add_basic_macros`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.add_basic_macros.rst) (flux-tunable defaults — see [Providers guide](providers.md#seeding-gate-macros-with-add_basic_macros)).

**Pulse caveat:** supported for **gate schedules only**. Qiskit Pulse **`Measure` / measurement instructions** are **not** supported. Use circuit-level `measure` + [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst) for readout in hybrid programs.

- **Guide:** [Backend — Pulse scope](backend.md#pulse-support-qiskit-1-x-legacy)
- **Example:** `examples/circuit_calibrations_pulse.py`

### 2.2 Custom gates via QMInstructionProperties (Qiskit 2.x)

1. Define a gate at the circuit level.
2. Write a QUA macro.
3. Register via [`QMInstructionProperties`](apidocs/stubs/qiskit_qm_provider.backend.QMInstructionProperties.rst).
4. Call `backend.update_target()`.

Keeps the Target and qm_qasm compiler in sync for both `backend.run()` and `quantum_circuit_to_qua`.

- **Guide:** [Backend — custom calibrations](backend.md#custom-calibrations-qiskit-2-x)

## 3. Primitives: Sampler and Estimator on QOP

[`QMSamplerV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerV2.rst) and [`QMEstimatorV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMEstimatorV2.rst) reuse QuAM Targets, stream parameters via [`InputType`](apidocs/stubs/qiskit_qm_provider.parameter_table.InputType.rst), and map shot budgets to QUA loops. **Classified counts only** — see [Primitives guide](primitives.md).

### 3.1 Generated QUA programs (and how to inspect them)

Every primitive job and `backend.run()` exposes the generated QUA `Program` on `job.program`:

```python
from qm import generate_qua_script
print(generate_qua_script(job.program))
```

See [Jobs](jobs.md) for the full job interface (`qm_job`, `pubs`, IQCC `run_data`, lifecycle).

End-to-end snippet:

```python
from qm import generate_qua_script
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_qm_provider import (
    QMProvider, QMSamplerV2, QMSamplerOptions,
    QMEstimatorV2, QMEstimatorOptions,
)

provider = QMProvider(state_folder_path="/path/to/quam/state")
backend = provider.get_backend()

qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
qc = transpile(qc, backend)

sampler = QMSamplerV2(backend=backend, options=QMSamplerOptions(default_shots=256))
sampler_job = sampler.run([qc])
print("=== Sampler ===")
print(generate_qua_script(sampler_job.program))

obs = SparsePauliOp.from_list([("Z", 1.0)])
estimator = QMEstimatorV2(backend=backend, options=QMEstimatorOptions())
estimator_job = estimator.run([(qc.remove_final_measurements(inplace=False), obs, [])])
print("=== Estimator ===")
print(generate_qua_script(estimator_job.program))

backend_job = backend.run(qc, shots=256)
print("=== backend.run() ===")
print(generate_qua_script(backend_job.program))
```

- **Guide:** [Primitives](primitives.md)
- **API:** [Primitives reference](apidocs/qm_primitives.rst)

## 4. Hybrid QUA/Qiskit programs (embedding circuits in QUA)

Treat Qiskit circuits as **building blocks** inside larger QUA programs:

1. Transpile a `QuantumCircuit`.
2. Inside `with program():`, call [`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) with [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) when needed.
3. Use **`comp.outputs`** for classical results — a local-only [`MeasurementOutcomeTable`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable.rst) of [`MeasurementRegisterField`](apidocs/stubs/qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField.rst) handles (one per classical register, plus `_bit0`, … for loose clbits). **`comp.outputs["c"]`** returns the QUA var; use **`comp.outputs.get_parameter("c")`** for the field handle and **`comp.outputs.state_ints["c"]`** / **`comp.outputs.streams["c"]`** for bulk accessors.

```python
from qm.qua import program, save

with program() as prog:
    comp = backend.quantum_circuit_to_qua(syndrome_circuit)
    save(comp.outputs.state_ints["ancilla"], comp.outputs.streams["ancilla"])
```

**Scope requirement:** QUA variable accessors (`comp.outputs["c"]`, `.state_ints`, `.streams`, runtime `ParameterTable[...]`) must be used inside `with program():`.

**Local-only:** `comp.outputs` is tracked in private measurement registries on [`ParameterPool`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterPool.rst) (weakref `iter_measurement_outcome_tables` / `iter_measurement_register_fields`), **not** in the runtime/OPNIC registry. It does not support OPNIC `stream_back()` / `fetch_from_opx()`. Bridge to the host manually via your transport tables (e.g. histogram into `RewardParams`, then `reward.stream_back()`).

**Name overlap:** Runtime input fields and measurement outputs may share the same string (e.g. both `"c"`). They are different objects — access inputs via your `ParameterTable` and measurements via `comp.outputs`. ``ParameterPool.lookup_runtime_parameter(name)`` (alias of ``_lookup_parameter_by_name``) returns runtime parameters only. Call ``ParameterPool.reset()`` to clear both runtime and measurement registries in a long-lived session.

**Re-compile identity:** each `quantum_circuit_to_qua` call returns a new `QuaCircuitCompilation` with **fresh** field objects. ``comp.rewire_outputs(qc, new_result)`` refreshes wiring on the same wrapper. Measurement fields cannot be attached to runtime tables.

**Breaking change:** ``comp.outputs.c.state_int`` is invalid — use ``comp.outputs.state_ints["c"]`` or ``comp.outputs.get_parameter("c").state_int``.

See [measurement_outputs.md](measurement_outputs.md) for the full locality model.

Legacy [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst) remains available and accepts either `QuaCircuitCompilation` or a raw `CompilationResult`.

Powerful for error correction, closed-loop calibration, and QUARC/OPNIC hybrid loops.

- **Guides:** [Backend](backend.md), [Parameter Table](parameter_table.md)
- **Example:** [Error-Correction Workflow](error_correction.md)

## 5. Error-correction workflow (overview)

Repeated cycles: encode → syndrome measure (Qiskit circuit) → stream syndrome → classical decode → push recovery params → apply recovery (Qiskit circuit). [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) and [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst) keep the classical-quantum boundary explicit.

- **Guide:** [Error-Correction Workflow](error_correction.md)

## 6. Qiskit Experiments + IQCC (with caveats)

`examples/iqcc_t1_experiment.py` shows T1 characterization with Qiskit Experiments on an IQCC backend. Before adopting this pattern broadly, read the [home-page callout](index.md#using-qiskit-experiments-with-this-provider):

**Batch vs real-time:** Experiments emit large batches of near-identical circuits (AWG-style preloading). QUA prefers one program with real-time loops and streaming. For calibration sweeps, consider [Qualibrate](https://qualibrate-docs.quantum-machines.co/) or [qua-libs](https://github.com/qua-platform/qua-libs). Use this provider to **compose** Qiskit circuits into real-time QUA programs when that is the right model.

**Counts only:** experiments needing raw I/Q or kerneled data will not work yet. Only classified 0/1 outcomes are supported.

**Positive framing:** the compiler's value is frictionless advanced QUA with Qiskit handling circuit synthesis, visualization, transpilation, and portability — not replacing QUA entirely with Qiskit.

## Related

- **API Reference:** [apidocs/qm](apidocs/qm.rst)
- **Examples folder:** [github.com/arthurostrauss/qiskit-qm-provider/tree/main/examples](https://github.com/arthurostrauss/qiskit-qm-provider/tree/main/examples)
