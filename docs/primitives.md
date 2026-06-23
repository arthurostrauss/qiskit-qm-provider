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

## Running on IQCC Cloud

When the backend is an IQCC cloud backend, `QMSamplerV2.run()` / `QMEstimatorV2.run()` return an `IQCCSamplerJob` / `IQCCEstimatorJob`. For streamed `input_type`s, the job submits the QUA program together with an auto-generated **sync hook** — a small Python script that runs on the cloud side and pushes the per-pub parameter values (and, for the Estimator, the observable indices) into the running program cycle-by-cycle.

**The sync hook is fully self-contained.** It imports only:

```python
from iqcc_cloud_client.runtime import get_qm_job
```

and drives the QM job directly. It does **not** require `qiskit_qm_provider`, `numpy`, or `qualang_tools` to be installed in the cloud runtime, so any IQCC user can run streamed primitives without extra packages. Parameter tables are serialised to plain Python data before submission and the push logic is rendered from a Jinja template; values are coerced to each parameter's QUA type (`int` / `fixed` / `bool`).

### Supported `input_type` on IQCC

| `input_type` | IQCC support | Cloud mechanism |
|--------------|--------------|-----------------|
| `INPUT_STREAM` | ✓ | `job.push_to_input_stream(name, value)` |
| `IO1` / `IO2` | ✓ | `job.set_io_values(io1=…)` with pause/resume synchronization |
| `None` | ✓ | Values bound at compile time (no sync hook); only for small parameter sets |
| `OPNIC` | ✗ | Raises `NotImplementedError` — QUARC-backed OPNIC transport is not yet available over the cloud sync hook |

```python
from qiskit_qm_provider import (
    IQCCProvider,
    QMSamplerV2,
    QMSamplerOptions,
    InputType,
    add_basic_macros,
)

# 1. Grab the cloud backend by name ("arbel", or whichever quantum computer you have access to).
backend = IQCCProvider().get_backend("arbel")

# 2. IQCC does not yet ship the standard single-qubit gate macros, so add them once
#    after fetching the backend. This populates x, sx, rz, sy, sydg, measure, reset,
#    delay, id, and cz macros and syncs the backend target.
add_basic_macros(backend)

# 3. Streamed primitives run on any IQCC user's cloud runtime — no provider install needed cloud-side.
sampler = QMSamplerV2(backend=backend, options=QMSamplerOptions(input_type=InputType.INPUT_STREAM))
job = sampler.run([(circuit, parameter_values)])

# OPNIC over IQCC is not supported yet and raises at run():
QMSamplerV2(backend=backend, options=QMSamplerOptions(input_type=InputType.OPNIC)).run([circuit])
# NotImplementedError: OPNIC input_type is not yet supported for IQCC cloud jobs; use INPUT_STREAM or IO1/IO2.
```

### IQCC cloud failures and `run_data`

Cloud-side failures (config validation, `open_qm` errors, etc.) often surface locally as a misleading `KeyError` on a measurement stream (for example `KeyError: '__c_0'`) because the QUA program never reached the streaming stage.

All IQCC wrapper jobs — [`IQCCJob`](apidocs/stubs/qiskit_qm_provider.job.qm_job.IQCCJob.rst), [`IQCCSamplerJob`](apidocs/stubs/qiskit_qm_provider.job.qm_sampler_job.IQCCSamplerJob.rst), and [`IQCCEstimatorJob`](apidocs/stubs/qiskit_qm_provider.job.qm_estimator_job.IQCCEstimatorJob.rst) — expose the raw IQCC execution record on **`job.run_data`** (backed by `job.qm_job._run_data`). Typical keys:

| Key | Content |
|-----|---------|
| `stdout` | QM / QOP log lines from the cloud runtime |
| `stderr` | Python traceback when the remote script failed |
| `result` | Timing and fridge metadata when execution completed |

When `job.result()` is called, the wrapper inspects `run_data["stderr"]` and, if it contains a Python traceback, raises [`IQCCCloudExecutionError`](apidocs/stubs/qiskit_qm_provider.job.iqcc_job_mixin.IQCCCloudExecutionError.rst) with the **exact cloud stderr** as the exception message — instead of a local stream `KeyError`.

```python
from qiskit_qm_provider.job import IQCCCloudExecutionError

try:
    result = job.result()
except IQCCCloudExecutionError as exc:
    print(exc)  # full cloud traceback
    print(job.run_data["stdout"])  # e.g. PHYSICAL CONFIG ERROR lines
```

You can also inspect `job.run_data` after submission without calling `result()` when debugging a failed cloud run.

## Debugging generated QUA

Every primitive job and `backend.run()` exposes the generated QUA `Program` on `job.program`. See the [Jobs guide](jobs.md) for the full property table (`qm_job`, `pubs`, IQCC `run_data`, …).

```python
from qm import generate_qua_script
print(generate_qua_script(job.program))
```

- **Guide:** [Jobs](jobs.md), [Workflows — Generated QUA programs](workflows.md#generated-qua-programs-and-how-to-inspect-them)

## Related

- **Guide:** [Workflows — Primitives on QOP](workflows.md#primitives-sampler-and-estimator-on-qop)
- **API:** [Primitives reference](apidocs/qm_primitives.rst), [Jobs reference](apidocs/qm_job.rst)
- **Examples:** `examples/sampler_workflow.py`, `examples/estimator_workflow.py`
