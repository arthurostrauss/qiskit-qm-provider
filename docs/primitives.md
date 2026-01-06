# Primitives

## QMEstimatorV2

`qiskit_qm_provider.primitives.qm_estimator.QMEstimatorV2`

A custom implementation of `BaseEstimatorV2` optimized for QOP.

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
