# Primitives

`QMSamplerV2` and `QMEstimatorV2` are **QOP-aware adapters** of Qiskit's V2 Sampler and Estimator — same pub interface, but execution leverages input streaming, real-time parameter updates, and (for Estimator) on-device control flow.

For signatures and options fields, see the [Primitives API reference](apidocs/qm_primitives.rst).

## Purpose

Generic cloud primitives assume parameters are bound at submission time. QOP workloads often **stream parameters cycle-by-cycle** via `InputType` (`INPUT_STREAM`, `IO1`, `IO2`, `DGX_Q`). These primitives expose that capability while reusing QuAM-derived Targets for transpilation.

The traditional `run` interface mimics Sampler-like behavior for users who prefer the classic backend API.

## Classified measurement outcomes only

**Only classified results (0/1 bitstrings / counts) are reliably supported today.** Experiments requiring raw I & Q, kerneled shots, or other non-discriminated outputs are **not yet supported**. Contributors welcome.

Although `QMSamplerOptions.meas_level` accepts `"kerneled"` and `"avg_kerneled"` in the API, those paths are **not production-ready**. Use `"classified"` (the default) for all current workflows, including Qiskit Experiments.

## QMSamplerV2

Shot-based measurement counts. Maps to backend `run()` under the hood. The returned job exposes the generated QUA program on `job.program`:

```python
from qm import generate_qua_script

sampler_job = sampler.run([qc])
print(generate_qua_script(sampler_job.program))
```

## QMEstimatorV2

Expectation values and standard errors with optional abelian grouping of observables. Estimator observables assume classified measurement outcomes.

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
| `input_type` | ✓ | ✓ | How parameters reach the OPX: stream, IO, DGX, or `None` (compile-time preload — only for small parameter sets) |
| `default_shots` | ✓ | — | Default shots when not specified in `run()` |
| `default_precision` | — | ✓ | Default precision (e.g. 1/√4096) when not specified |
| `abelian_grouping` | — | ✓ | Group commuting observables (default `True`) |
| `run_options` | ✓ | ✓ | Extra dict passed through to `backend.run()` |
| `meas_level` | ✓ | — | **Use `"classified"` only** — see limitation above |

Set `input_type=None` to bind all parameter values at compile time (suitable only when the number of distinct parameter sets is small).

## Debugging generated QUA

- **Guide:** [Workflows — Generated QUA programs](workflows.md#generated-qua-programs-and-how-to-inspect-them)

## Related

- **Guide:** [Workflows — Primitives on QOP](workflows.md#primitives-sampler-and-estimator-on-qop)
- **API:** [Primitives reference](apidocs/qm_primitives.rst), [Jobs reference](apidocs/qm_job.rst)
- **Examples:** `examples/sampler_workflow.py`, `examples/estimator_workflow.py`
