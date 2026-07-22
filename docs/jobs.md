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
| `cancel()` | ✓ | ✓ | Forwards to all underlying QM SDK jobs |
| `result()` | ✓ | ✓ | Builds Qiskit result from streamed data |
| `get_qm_job(idx)` | ✓ | ✓ | Return the QM SDK job at *idx* (default: `0`) |
| `get_program(idx)` | ✓ | ✓ | Return the compiled `Program` at *idx* (default: `0`) |
| `get_result_handles(idx)` | ✓ | ✓ | Return the result-handles fetcher at *idx* (default: `0`) |

[`IQCCJob`](apidocs/stubs/qiskit_qm_provider.job.qm_job.IQCCJob.rst) does **not** implement `status()` — poll via IQCC cloud APIs or inspect [`run_data`](apidocs/stubs/qiskit_qm_provider.job.iqcc_job_mixin.IQCCJobMixin.rst).

`IQCCSamplerJob` and `IQCCEstimatorJob` **do** implement `status()` via `IQCCJobMixin`, but with important caveats:

> IQCC execution is **synchronous** — `submit()` blocks until the remote OPX program completes, so `CloudJob.status` is unconditionally `"completed"` and carries no failure information.  `status()` therefore ignores `CloudJob.status` entirely and inspects `_run_data["stderr"]` instead: it returns `JobStatus.ERROR` when the cloud runtime's stderr contains a Python traceback, `JobStatus.DONE` otherwise.  Call `result()` to get the full [`IQCCCloudExecutionError`](apidocs/stubs/qiskit_qm_provider.job.iqcc_job_mixin.IQCCCloudExecutionError.rst) with the traceback text.

## Chunked execution (`max_circuits`)

When the number of circuits or PUBs exceeds `max_circuits`, the provider automatically splits them into consecutive QUA programs and executes each chunk in order.  Results are stitched back transparently — callers see a single `Result` / `PrimitiveResult` regardless of how many chunks were used.

```python
# Run 50 PUBs with a hardware limit of 20 circuits per program
sampler = QMSamplerV2(backend)
job = sampler.run(pubs, max_circuits=20)   # produces ceil(50/20) = 3 QUA programs
result = job.result()                      # stitched PrimitiveResult with 50 entries
```

The compiled programs are accessible on `job.programs` (a list) and via `get_program(idx)`:

```python
print(f"{len(job.programs)} QUA programs generated")
for i, prog in enumerate(job.programs):
    print(f"=== chunk {i} ===")
    print(generate_qua_script(prog))
```

Stream keys inside each chunk are **local** (0-based within the chunk), not global.  The job tracks the mapping internally via `compute_locator()` and translates global pub/circuit indices to the correct chunk and local key during result assembly.

### Standard QM vs IQCC output shape

The `bit_array_from_measurement_stream` helper in `stream_assembly.py` handles both backends:

- **Standard QM / SaaS** — `fetch_all()` returns a flat array; `reshape` normalises it to `(*pub.shape, shots)`.
- **IQCC cloud** — `fetch_all()` may return an array with an extra leading dimension; the same `reshape` collapses it to the expected shape, provided the total element count matches.

Both paths reach the same `BitArray` shape so downstream code is backend-agnostic.

## Inspecting the generated QUA program

All job types expose the compiled QUA programs on **`job.programs`** — always a `list[`[`qm.Program`](https://docs.quantum-machines.co/latest/docs/qm/quam/user_guide/qua/program/)`]`, length 1 when no chunking occurred. Use `get_program()` to fetch the single program (index 0) without worrying about list indexing:

```python
from qm import generate_qua_script

job = sampler.run([qc])
print(generate_qua_script(job.get_program()))
```

The same works for estimator and `backend.run()` jobs:

```python
backend_job = backend.run(qc, shots=256)
estimator_job = estimator.run([(circuit, observables, param_values)])

print("=== backend.run() ===")
print(generate_qua_script(backend_job.get_program()))

print("=== Estimator ===")
print(generate_qua_script(estimator_job.get_program()))
```

`job.programs` is available **immediately after** the job object is constructed (before or after `submit()`), because compilation happens at job creation time.

For chunked jobs (multiple programs), iterate the list or fetch a specific chunk by index:

```python
# Iterate all chunks:
for chunk_idx, prog in enumerate(job.programs):
    print(f"=== chunk {chunk_idx} ===")
    print(generate_qua_script(prog))

# Or fetch a specific chunk:
print(generate_qua_script(job.get_program(2)))  # third chunk
```

## Properties and attributes

### Shared across job types

| Name | Type | When set | Purpose |
|------|------|----------|---------|
| `job_id` | `str` | After `submit()` | QM SDK job id (comma-separated for multi-program jobs) |
| `backend` | [`QMBackend`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) | Construction | Backend that compiled the circuits |
| `metadata` | `dict` | Construction | Run options forwarded from backend/primitive (`compiler_options`, `simulate`, `timeout`, …) |
| `programs` | `list[Program]` | Construction | Compiled QUA programs — always a list; use with `generate_qua_script` |
| `qm_jobs` | `list[RunningQmJob \| QmPendingJob]` | After `submit()` | All underlying QM SDK handles — always a list, length 1 for non-chunked execution |
| `result_handles` | `list[QM result fetcher]` | After `submit()` | One fetcher per submitted job |
| `get_qm_job(idx=None)` | `RunningQmJob \| QmPendingJob` | After `submit()` | Return the QM SDK job at *idx* (default `0`). Shorthand for `qm_jobs[idx]` |
| `get_program(idx=None)` | `Program` | Construction | Return the compiled program at *idx* (default `0`). Shorthand for `programs[idx]` |
| `get_result_handles(idx=None)` | QM result fetcher | After `submit()` | Return the fetcher at *idx* (default `0`). Shorthand for `result_handles[idx]` |

`QMJob` also stores **`qm`** — the [`QuantumMachine`](https://docs.quantum-machines.co/latest/docs/qm/quam/user_guide/quam/components/quam_root/qmmachine/) (or cloud equivalent) used for execution.

### Primitive jobs (`QMSamplerJob`, `QMEstimatorJob`)

| Name | Type | Purpose |
|------|------|---------|
| `pubs` | `list[SamplerPub \| EstimatorPub]` | PUBs passed to `run()` |
| `inputs` | `dict` | Snapshot of pubs, `input_type`, and `metadata` |
| `programs` | `list[Program]` | Same as above — sampler/estimator QUA programs |
| `result_handles` | `list[QM result fetcher]` | One fetcher per submitted job (via [`QMPrimitiveJob`](apidocs/stubs/qiskit_qm_provider.job.qm_primitive_job.QMPrimitiveJob.rst)); use `get_result_handles()` for single-job access |
| `runtime_pubs` *(estimator only)* | `list[_ExecutionPlan]` | Compiled execution plans — one per EstimatorPub; exposes observable grouping, parameter tables, and the switch-statement data streamed to the OPX |

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

For streamed primitives, parameter values are pushed **after** submit via the parameter tables bound at compile time. Use the getter methods to access the live handles:

```python
job = sampler.run([(qc, param_values)])
# submit already ran

job.get_result_handles()            # stream handle for the first (only) job
param_table.push_to_opx(..., job=job.get_qm_job())
# `qm=backend.qm` is optional legacy back-compat; current QUA drives IO via the job.
```

On IQCC, streamed jobs auto-generate a **sync hook** script that performs this pushing on the cloud side (see [Primitives](primitives.md#running-on-iqcc-cloud)).

## Debugging checklist

1. **Print the QUA** — `print(generate_qua_script(job.get_program()))` (or iterate `job.programs` for chunked jobs)
2. **Check job id** — `job.job_id` after submit
3. **Poll status** — `job.status()` (not on `IQCCJob`; on `IQCCSamplerJob` / `IQCCEstimatorJob` this checks stderr, not the QM SDK status property)
4. **IQCC failures** — `job.run_data` before trusting `result()`
5. **Stream keys** — `job.get_result_handles()` for the active stream fetcher

## Related

- **Guide:** [Primitives](primitives.md), [Workflows — generated QUA](workflows.md#generated-qua-programs-and-how-to-inspect-them)
- **API:** [Jobs reference](apidocs/qm_job.rst), [Primitives reference](apidocs/qm_primitives.rst)
