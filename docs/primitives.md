---
title: Primitives
nav_order: 5
parent: Home
---

# Primitives

## QMEstimatorV2

`qiskit_qm_provider.primitives.qm_estimator.QMEstimatorV2`

A custom implementation of `BaseEstimatorV2` optimized for QOP.

### Generated QUA program (debugging)

`QMEstimatorV2` automatically generates the underlying QUA program required to run your pubs on
QOP. If you need to inspect what was generated, the returned job exposes the QUA `Program` on
`job.program`, and you can print it as a QUA script via:

```python
from qm import generate_qua_script
print(generate_qua_script(job.program))
```

See the full end-to-end snippet in
[Workflows and Examples](workflows.md#31-generated-qua-programs-and-how-to-inspect-them).

### `__init__(backend: QMBackend, options: QMEstimatorOptions | dict | None = None)`
- `backend`: The QMBackend.
- `options`: Options for the estimator.

### `run(pubs: Iterable[EstimatorPubLike], *, precision: float | None = None)`
Runs the estimator.

## QMEstimatorOptions

- `default_precision`: Default precision (1/sqrt(shots)).
- `abelian_grouping`: Group commuting observables (Default: True).
- `input_type`: Mechanism for parameter loading (`InputType.INPUT_STREAM`, `DGX_Q`, etc.).
- `run_options`: Dictionary of options passed to `backend.run`.

---

## QMSamplerV2

`qiskit_qm_provider.primitives.qm_sampler.QMSamplerV2`

A custom implementation of `BaseSamplerV2`.

### Generated QUA program (debugging)

`QMSamplerV2` automatically generates the underlying QUA program required to run your pubs on
QOP. If you need to inspect what was generated, the returned job exposes the QUA `Program` on
`job.program`, and you can print it as a QUA script via:

```python
from qm import generate_qua_script
print(generate_qua_script(job.program))
```

See the full end-to-end snippet in
[Workflows and Examples](workflows.md#31-generated-qua-programs-and-how-to-inspect-them).

### `__init__(backend: QMBackend, options: QMSamplerOptions | dict | None = None)`
- `backend`: The QMBackend.
- `options`: Options for the sampler.

### `run(pubs: Iterable[SamplerPubLike], *, shots: int | None = None)`
Runs the sampler.

## QMSamplerOptions

- `default_shots`: Default number of shots (Default: 1024).
- `input_type`: Mechanism for parameter loading.
- `run_options`: Dictionary of options passed to `backend.run`.
- `meas_level`: "classified", "kerneled", or "avg_kerneled".
