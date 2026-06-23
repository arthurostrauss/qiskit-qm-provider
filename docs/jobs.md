# Jobs

Every execution path in this provider — [`QMBackend.run()`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst), [`QMSamplerV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerV2.rst), and [`QMEstimatorV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMEstimatorV2.rst) — returns a **job handle** immediately. The handle exposes the compiled QUA [`Program`](https://docs.quantum-machines.co/latest/docs/qm/quam/user_guide/qua/program/) for inspection, bridges to the underlying QM SDK job, and builds Qiskit results when you call [`result()`](apidocs/stubs/qiskit_qm_provider.job.QMJob.rst).

For method signatures, see the [Jobs API reference](apidocs/qm_job.rst).

## Job types

| Class | Returned by | `result()` type |
|-------|-------------|-----------------|
| [`QMJob`](apidocs/stubs/qiskit_qm_provider.job.QMJob.rst) | `backend.run()` (local / SaaS QMM) | [`qiskit.result.Result`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.result.Result) |
| [`IQCCJob`](apidocs/stubs/qiskit_qm_provider.job.qm_job.IQCCJob.rst) | `backend.run()` (IQCC cloud) | Same as `QMJob`; cloud failures surface via [`run_data`](apidocs/stubs/qiskit_qm_provider.job.iqcc_job_mixin.IQCCJobMixin.rst) |
| [`QMSamplerJob`](apidocs/stubs/qiskit_qm_provider.job.QMSamplerJob.rst) | `QMSamplerV2.run()` | [`PrimitiveResult`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.PrimitiveResult) of [`SamplerPubResult`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.SamplerPubResult) |
| [`QMEstimatorJob`](apidocs/stubs/qiskit_qm_provider.job.QMEstimatorJob.rst) | `QMEstimatorV2.run()` | [`PrimitiveResult`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.PrimitiveResult) of estimator pub results |
| [`IQCCSamplerJob`](apidocs/stubs/qiskit_qm_provider.job.qm_sampler_job.IQCCSamplerJob.rst) | `QMSamplerV2.run()` on IQCC | Same as `QMSamplerJob` |
| [`IQCCEstimatorJob`](apidocs/stubs/qiskit_qm_provider.job.qm_estimator_job.IQCCEstimatorJob.rst) | `QMEstimatorV2.run()` on IQCC | Same as `QMEstimatorJob` |

Primitive jobs inherit [`QMPrimitiveJob`](apidocs/stubs/qiskit_qm_provider.job.qm_primitive_job.QMPrimitiveJob.rst). IQCC wrapper jobs mix in [`IQCCJobMixin`](apidocs/stubs/qiskit_qm_provider.job.iqcc_job_mixin.IQCCJobMixin.rst) for cloud diagnostics.

## Typical lifecycle

```python
job = backend.run(qc, shots=1024)          # or sampler.run([pub]) / estimator.run([pub])
# job is submitted automatically for backend.run() and primitives

print(job.job_id)                          # QM SDK id after submit (was "pending" briefly)
print(job.status())                        # JobStatus.RUNNING / DONE / …

result = job.result()                      # blocks until streams are complete
```

| Method | `QMJob` | `QMPrimitiveJob` | Notes |
|--------|---------|------------------|-------|
| `submit()` | Called inside `from_circuits` / manual | Called by primitive `run()` | Re-submit raises on primitives |
| `status()` | ✓ | ✓ | Maps QM SDK status to [`JobStatus`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.JobStatus) |
| `done()` / `running()` / `cancelled()` | — | ✓ | Convenience wrappers around `status()` |
| `cancel()` | ✓ | ✓ | Forwards to underlying `qm_job` |
| `result()` | ✓ | ✓ | Builds Qiskit result from streamed data |

[`IQCCJob`](apidocs/stubs/qiskit_qm_provider.job.qm_job.IQCCJob.rst) does **not** implement `status()` — poll via IQCC cloud APIs or inspect [`run_data`](apidocs/stubs/qiskit_qm_provider.job.iqcc_job_mixin.IQCCJobMixin.rst).

## Inspecting the generated QUA program

All job types expose the compiled QUA programs on **`job.programs`** — always a `list[`[`qm.Program`](https://docs.quantum-machines.co/latest/docs/qm/quam/user_guide/qua/program/)`]`, length 1 when no chunking occurred. Use the QM SDK helper to print human-readable QUA source:

```python
from qm import generate_qua_script

job = sampler.run([qc])
print(generate_qua_script(job.programs[0]))
```

The same works for estimator and `backend.run()` jobs:

```python
backend_job = backend.run(qc, shots=256)
estimator_job = estimator.run([(circuit, observables, param_values)])

print("=== backend.run() ===")
print(generate_qua_script(backend_job.programs[0]))

print("=== Estimator ===")
print(generate_qua_script(estimator_job.programs[0]))
```

`job.programs` is available **immediately after** the job object is constructed (before or after `submit()`), because compilation happens at job creation time.

For chunked jobs (multiple programs), iterate the list:

```python
for chunk_idx, prog in enumerate(job.programs):
    print(f"=== chunk {chunk_idx} ===")
    print(generate_qua_script(prog))
```

## Properties and attributes

### Shared across job types

| Name | Type | When set | Purpose |
|------|------|----------|---------|
| `job_id` | `str` | After `submit()` | QM SDK job id (comma-separated for multi-program `QMJob`) |
| `backend` | [`QMBackend`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) | Construction | Backend that compiled the circuits |
| `metadata` | `dict` | Construction | Run options forwarded from backend/primitive (`compiler_options`, `simulate`, `timeout`, …) |
| `programs` | `list[Program]` | Construction | Compiled QUA programs — always a list; use with `generate_qua_script` |
| `qm_job` | `RunningQmJob` / `QmPendingJob` / `list` | After `submit()` | Low-level QM SDK handle (`cancel`, `push_to_input_stream`, …) |
| `result_handles` | QM result fetcher (or `list`) | After `submit()` | Same as `qm_job.result_handles` — stream keys for debugging |

`QMJob` also stores **`qm`** — the [`QuantumMachine`](https://docs.quantum-machines.co/latest/docs/qm/quam/user_guide/quam/components/quam_root/qmmachine/) (or cloud equivalent) used for execution.

### Primitive jobs (`QMSamplerJob`, `QMEstimatorJob`)

| Name | Type | Purpose |
|------|------|---------|
| `pubs` | `list[SamplerPub \| EstimatorPub]` | PUBs passed to `run()` |
| `inputs` | `dict` | Snapshot of pubs, `input_type`, and `metadata` |
| `programs` | `list[Program]` | Same as above — sampler/estimator QUA programs |
| `result_handles` | QM result fetcher | `qm_job.result_handles` after submit (via [`QMPrimitiveJob`](apidocs/stubs/qiskit_qm_provider.job.qm_primitive_job.QMPrimitiveJob.rst)) |

### IQCC wrapper jobs

| Name | Type | Purpose |
|------|------|---------|
| `run_data` | `dict \| None` | Raw IQCC cloud record (`stdout`, `stderr`, `result` timing metadata) |

When the remote runtime fails, `result()` raises [`IQCCCloudExecutionError`](apidocs/stubs/qiskit_qm_provider.job.iqcc_job_mixin.IQCCCloudExecutionError.rst) with the cloud traceback instead of a misleading local stream error. See [Primitives — IQCC cloud failures](primitives.md#iqcc-cloud-failures-and-run_data).

```python
from qiskit_qm_provider.job import IQCCCloudExecutionError

try:
    result = job.result()
except IQCCCloudExecutionError as exc:
    print(exc)                    # cloud stderr / traceback
    print(job.run_data["stdout"]) # QM log lines from the remote runtime
```

## Pushing data on a running job

For streamed primitives, parameter values are pushed **after** submit via the parameter tables bound at compile time. Internally the job uses `qm_job`:

```python
job = sampler.run([(qc, param_values)])
# submit already ran; qm_job is live

job.result_handles                 # stream handles (all job types)
param_table.push_to_opx(..., job=job.qm_job, qm=backend.qm)
```

On IQCC, streamed jobs auto-generate a **sync hook** script that performs this pushing on the cloud side (see [Primitives](primitives.md#running-on-iqcc-cloud)).

## Debugging checklist

1. **Print the QUA** — `print(generate_qua_script(job.programs[0]))` (or iterate `job.programs` for chunked jobs)
2. **Check job id** — `job.job_id` after submit
3. **Poll status** — `job.status()` (not on `IQCCJob`)
4. **IQCC failures** — `job.run_data` before trusting `result()`
5. **Stream keys** — `job.result_handles` for raw stream names

## Related

- **Guide:** [Primitives](primitives.md), [Workflows — generated QUA](workflows.md#generated-qua-programs-and-how-to-inspect-them)
- **API:** [Jobs reference](apidocs/qm_job.rst), [Primitives reference](apidocs/qm_primitives.rst)
