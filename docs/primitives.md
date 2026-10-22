# Primitives

[`QMSamplerV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerV2.rst) and [`QMEstimatorV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMEstimatorV2.rst) are **QOP-aware adapters** of Qiskit's V2 Sampler and Estimator — same pub interface, but execution leverages input streaming, real-time parameter updates, and (for Estimator) on-device control flow.

For signatures and options fields, see the [Primitives API reference](apidocs/qm_primitives.rst).

## Purpose

Generic cloud primitives assume parameters are bound at submission time. QOP workloads often **stream parameters cycle-by-cycle** via [`InputType`](apidocs/stubs/qiskit_qm_provider.parameter_table.InputType.rst) (`INPUT_STREAM`, `IO1`, `IO2`, `OPNIC`). These primitives expose that capability while reusing QuAM-derived Targets for transpilation.

The traditional [`QMBackend.run()`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) interface mimics Sampler-like behavior for users who prefer the classic backend API. Like `backend.run`, primitives respect the [`max_circuits`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) backend option and will split large batches into multiple QUA programs when needed. See [Backend — multi-circuit batches](backend.md#multi-circuit-batches-and-max_circuits).

## Classified measurement outcomes only

**Only classified results (0/1 bitstrings / counts) are reliably supported today.** Experiments requiring raw I & Q, kerneled shots, or other non-discriminated outputs are **not yet supported**. Contributors welcome.

Although [`QMSamplerOptions.meas_level`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerOptions.rst) accepts `"kerneled"` and `"avg_kerneled"` in the API, those paths are **not production-ready**. Use `"classified"` (the default) for all current workflows, including Qiskit Experiments.

## QMSamplerV2

Shot-based measurement counts. Maps to backend `run()` under the hood. The returned [`QMSamplerJob`](apidocs/stubs/qiskit_qm_provider.job.QMSamplerJob.rst) exposes the compiled QUA programs via `get_program()`:

```python
from qm import generate_qua_script

sampler = QMSamplerV2(backend=backend, options=QMSamplerOptions(default_shots=256))
sampler_job = sampler.run([qc])
print(generate_qua_script(sampler_job.get_program()))
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

### Which qubits are measured

Circuits and observables are taken as ISA: observables span the full physical register, as in Qiskit (transpile the circuit, then call `observable.apply_layout(circuit.layout)`). The estimator measures a qubit **if and only if some term of some observable in the PUB acts non-trivially on it**. The set is the union over the PUB's whole observables array, because one compiled circuit serves every observable and binding of the PUB; on the other qubits every observable is the identity, so they cannot change any expectation value.

- A qubit that no gate touches is still measured if an observable acts on it.
- Ancillas are treated like any other qubit: `apply_layout` pads them with identity, so they are normally not measured, but an observable that acts on one is measured faithfully.
- Qubits that carry gates but no observable support are not measured, which saves their basis switch and readout.
- If every observable is the identity, one qubit is measured (the first qubit of the circuit's layout), as in Qiskit's `BackendEstimatorV2`.

Per measured qubit `q_j`, the prepared circuit has a basis switch driven by an input `obs_j` (0: none, 1: `h`, 2: `sdg; h`) followed by a measurement into bit `j` of a `__c` register. `EstimatorPlan.from_pub` requires this readout (it rejects circuits without it, and observables acting on a qubit outside `__c`); `QMEstimatorV2.run` adds it through `validate_estimator_pubs`. A PUB with an empty bindings array is rejected.

### How observables reach the OPX

Each measured qubit gets one basis code per measurement group: 0 for Z or I, 1 for X, 2 for Y. These codes are packed 2 bits per qubit, 15 qubits per 32-bit word. For each parameter binding, the program receives one int row, `[number of groups, packed words...]`, and unpacks it in QUA with shifts and masks.

- **Input streams:** one push per binding. A row shared by every binding is sent once.
- **IO1/IO2:** only the used part of the row is exchanged.
- **Compile time (`input_type=None`):** the same rows are preloaded into the program.

## Parameter values and fixed-point range

Every circuit parameter reaches the program as a QUA `fixed`: a signed 4.28 fixed-point number, with range **[-8, 8)** and resolution 2⁻²⁸ ≈ 3.7·10⁻⁹. This holds for every `input_type`, including compile-time preloading. Values outside this range are **not supported**. When a `QMSamplerV2` or `QMEstimatorV2` job is created, any PUB with a bound value outside the range (or a NaN or infinite value) is rejected with a `ValueError` that names the PUB, the parameter and the binding. Nothing is clipped or wrapped for you.

Angles in [-π, π] or [0, 2π] fit. Quantities with a larger natural scale do not, for example durations, frequencies or angles accumulated over many turns. Bind such a quantity in a rescaled unit that stays inside [-8, 8), and undo the scale in the circuit, as long as the expression stays in range (see the warning below).

```{warning}
The range check covers the **bound values only**. It does not cover expressions built from parameters. A gate argument such as `rx(3 * theta)`, `rz(theta + phi + chi)` or `rx(theta * phi)`, or a backend macro that scales an angle, is evaluated **in real time, in fixed-point arithmetic, on the controller**. Each value can be in range while an intermediate or final result is not: with `theta = 3.0`, `3 * theta = 9.0` cannot be represented in `fixed`, and the provider raises no error. The gate then receives a wrong value. Products of small values also lose resolution, since only 28 fractional bits remain.

Before running, check the extreme values of every parameter expression in your circuits, not just of the parameters.
```

When the risk is high for your use case (large dynamic ranges, long sums, products of parameters), please reach out to the maintainer first (open an issue on the [GitHub repository](https://github.com/arthurostrauss/qiskit-qm-provider/issues) or write to the author listed in `pyproject.toml`). In general, such kernels have to be written in QUA as real-time processing with explicit rescaling, rather than expressed as Qiskit parameter arithmetic. QUA's `Cast` utilities (`from qm.qua import Cast`) are the main tools for this:

- `Cast.mul_fixed_by_int(x, n)` and `Cast.mul_int_by_fixed(n, x)` multiply a `fixed` by an `int` without first converting the integer to `fixed`. An integer ≥ 8 cannot be a `fixed`, so carry large integer factors as `int` and combine them only through these helpers. The result must still fit its output type.
- `Cast.to_int`, `Cast.to_fixed` and `Cast.to_bool` convert values between types.
- `Cast.unsafe_cast_int`, `Cast.unsafe_cast_fixed` and `Cast.unsafe_cast_bool` reinterpret the raw 32 bits without conversion. Use them with integer shifts to scale by powers of two, or to carry a wider-range quantity as an integer part and a fractional part.

See the [QUA reference guide](https://docs.quantum-machines.co/latest/docs/Guides/qua_ref/) in the QM documentation for the full semantics of QUA variable types, arithmetic and the `Cast` module.

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

## Large PUB batches and `max_circuits`

When the number of PUBs passed to `sampler.run()` or `estimator.run()` exceeds `backend.options.max_circuits` (default `30`), the provider automatically splits them into consecutive chunks. Each chunk is compiled into its own QUA program and queued sequentially on QOP. The returned `PrimitiveResult` always has one entry per input PUB **in the original order** — the splitting is fully transparent to calling code.

```python
# Reduce the limit so that even small batches trigger splitting:
backend.set_options(max_circuits=5)

sampler = QMSamplerV2(backend=backend)
# 12 PUBs -> 3 programs of 5, 5, 2 PUBs queued sequentially
job = sampler.run([pub_0, pub_1, ..., pub_11])
result = job.result()  # 12 SamplerPubResult entries, in original order
```

Set `max_circuits=None` to disable splitting and always build a single program regardless of batch size.

## Running on IQCC Cloud

When the backend is an IQCC cloud backend, `QMSamplerV2.run()` / `QMEstimatorV2.run()` return an `IQCCSamplerJob` / `IQCCEstimatorJob`. For streamed `input_type`s, the job submits the QUA program together with an auto-generated **sync hook** — a small Python script that runs on the cloud side and pushes, in program order, each PUB's parameter rows (and, for the Estimator, its packed observable rows) into the running program.

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

All IQCC wrapper jobs — [`CloudQMJob`](apidocs/stubs/qiskit_qm_provider.job.qm_job.CloudQMJob.rst), [`IQCCSamplerJob`](apidocs/stubs/qiskit_qm_provider.job.qm_sampler_job.IQCCSamplerJob.rst), and [`IQCCEstimatorJob`](apidocs/stubs/qiskit_qm_provider.job.qm_estimator_job.IQCCEstimatorJob.rst) — expose the raw IQCC execution record on **`job.run_data`** (backed by `job.get_qm_job()._run_data`). Typical keys:

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

Every primitive job and `backend.run()` exposes the compiled QUA programs via `get_program()` (single/default) or by iterating `job.programs` (all chunks). See the [Jobs guide](jobs.md) for the full accessor table (`qm_jobs`, `get_qm_job()`, `get_program()`, `get_result_handles()`, `pubs`, IQCC `run_data`, …).

Both primitive jobs also expose their execution plans on **`job.runtime_pubs`**: one `SamplerPlan` or `EstimatorPlan` (from `qiskit_qm_provider.job.primitive_plans`) per input PUB. An `EstimatorPlan` shows the measured qubits, how observables were grouped (`metadata`, `obs_codes`) and the packed rows streamed to the OPX (`obs_rows`).

```python
from qm import generate_qua_script

# Non-chunked (default): use the getter
print(generate_qua_script(job.get_program()))

# Chunked: iterate all programs
for chunk_idx, prog in enumerate(job.programs):
    print(f"=== QUA program {chunk_idx} ===")
    print(generate_qua_script(prog))
```

- **Guide:** [Jobs](jobs.md), [Workflows — Generated QUA programs](workflows.md#generated-qua-programs-and-how-to-inspect-them)

## Related

- **Guide:** [Workflows — Primitives on QOP](workflows.md#primitives-sampler-and-estimator-on-qop)
- **API:** [Primitives reference](apidocs/qm_primitives.rst), [Jobs reference](apidocs/qm_job.rst)
- **Examples:** `examples/sampler_workflow.py`, `examples/estimator_workflow.py`
