# Primitives

[`QMSamplerV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerV2.rst) and [`QMEstimatorV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMEstimatorV2.rst) are **QOP-aware adapters** of Qiskit's V2 Sampler and Estimator — same pub interface, but execution leverages input streaming, real-time parameter updates, and (for Estimator) on-device control flow.

For signatures and options fields, see the [Primitives API reference](apidocs/qm_primitives.rst).

## Purpose

Generic cloud primitives assume parameters are bound at submission time. QOP workloads often **stream parameters cycle-by-cycle** via [`InputType`](apidocs/stubs/qiskit_qm_provider.parameter_table.InputType.rst) (`INPUT_STREAM`, `IO1`, `IO2`, `OPNIC`). These primitives expose that capability while reusing QuAM-derived Targets for transpilation.

The traditional [`QMBackend.run()`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) interface mimics Sampler-like behavior for users who prefer the classic backend API.

## Classified measurement outcomes only

**Only classified results (0/1 bitstrings / counts) are reliably supported today.** Experiments requiring raw I & Q, kerneled shots, or other non-discriminated outputs are **not yet supported**. Contributors welcome.

Although [`QMSamplerOptions.meas_level`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerOptions.rst) accepts `"kerneled"` and `"avg_kerneled"` in the API, those paths are **not production-ready**. Use `"classified"` (the default) for all current workflows, including Qiskit Experiments.

## QMSamplerV2

Shot-based measurement counts. Maps to backend `run()` under the hood. The returned [`QMSamplerJob`](apidocs/stubs/qiskit_qm_provider.job.QMSamplerJob.rst) exposes the generated QUA program on `job.program`:

```python
from qm import generate_qua_script

sampler = QMSamplerV2(backend=backend, options=QMSamplerOptions(default_shots=256))
sampler_job = sampler.run([qc])
print(generate_qua_script(sampler_job.program))
```

## QMEstimatorV2

Expectation values and standard errors with optional abelian grouping of observables. Estimator observables assume classified measurement outcomes. Returns a [`QMEstimatorJob`](apidocs/stubs/qiskit_qm_provider.job.QMEstimatorJob.rst).

```python
from qiskit_qm_provider import QMEstimatorV2, QMEstimatorOptions, InputType

options = QMEstimatorOptions(input_type=InputType.INPUT_STREAM)
estimator = QMEstimatorV2(backend=backend, options=options)
job = estimator.run([(circuit, observables, parameter_values)])
result = job.result()
```

## Options at a glance

| Option | Sampler | Estimator | Meaning |
|--------|---------|-----------|---------|
| `input_type` | ✓ | ✓ | How parameters reach the OPX: stream, IO, QUARC-backed OPNIC, or `None` (compile-time preload — only for small parameter sets) |
| `default_shots` | ✓ | — | Default shots when not specified in `run()` |
| `default_precision` | — | ✓ | Default precision (e.g. 1/√4096) when not specified |
| `abelian_grouping` | — | ✓ | Group commuting observables (default `True`) |
| `run_options` | ✓ | ✓ | Extra dict passed through to `backend.run()` |
| `meas_level` | ✓ | — | **Use `"classified"` only** — see limitation above |

Configure options via [`QMSamplerOptions`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerOptions.rst) or [`QMEstimatorOptions`](apidocs/stubs/qiskit_qm_provider.primitives.QMEstimatorOptions.rst). Set `input_type=None` to bind all parameter values at compile time (suitable only when the number of distinct parameter sets is small).

## Debugging generated QUA

Every primitive job and `backend.run()` exposes the generated QUA `Program` on `job.program`:

```python
from qm import generate_qua_script
print(generate_qua_script(job.program))
```

- **Guide:** [Workflows — Generated QUA programs](workflows.md#generated-qua-programs-and-how-to-inspect-them)

## Related

- **Guide:** [Workflows — Primitives on QOP](workflows.md#primitives-sampler-and-estimator-on-qop)
- **API:** [Primitives reference](apidocs/qm_primitives.rst), [Jobs reference](apidocs/qm_job.rst)
- **Examples:** `examples/sampler_workflow.py`, `examples/estimator_workflow.py`
