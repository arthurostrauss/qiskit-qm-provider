# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-10-29

### Added

- **Support for the `full_real_time_expr` Qiskit branch** — the provider now works with classical arrays from the author's Qiskit branch [`arthurostrauss/qiskit@full_real_time_expr`](https://github.com/arthurostrauss/qiskit/tree/full_real_time_expr), which is not merged upstream. The branch adds dynamic `for`-loop ranges (`expr.Range`), 1-D classical arrays (`types.Array`, indexed with `expr.index`), and a fix that lets `for`-loop bodies with a `Var` counter capture variables from the enclosing scope. See the new *Real-Time Classical Expressions* guide in the docs, which includes a two-qubit randomized-benchmarking example. On the branch, `expr.Range` uses OpenQASM 3 inclusive-stop semantics. Array inputs need qm-qasm ≥ 1.8.0. Everything below is inactive on stock Qiskit.
- **`export_vector_as_array` backend option** (default `True`) — the backend's OpenQASM 3 exporter emits each `ParameterVector` as one full-length `input array[float[64], N]`, through the `full_real_time_expr` branch's `qiskit.qasm3.ExperimentalFeatures.PARAMETER_VECTORS_AS_ARRAYS`. Ignored on stock Qiskit, which has no such feature and exports one scalar input per used element. Set it to `False` to export vectors as scalar inputs on the branch too (e.g. to read the program back with Qiskit's OpenQASM 3 importer, which cannot read `input` arrays).
- **`ParameterTable.from_qiskit` array inputs** — a classical input of type `types.Array(Uint | Bool | Float, n)` becomes a single array-valued `Parameter` (a QUA `int` / `bool` / `fixed` array of length `n`). Earlier versions misread it as a scalar `fixed`.
- **Unsupported classical types rejected** — Duration/Stretch variables, stores and inputs, `Array(Duration, n)` and zero-size arrays now raise one consistent `ValueError` in `ParameterTable.from_qiskit` and `quantum_circuit_to_qua` (durations are compile-time only in QUA; use them in delays). Previously a Duration input silently became a `fixed`.
- **Conditional pulse plays (`conditional_play`)** — apply a fixed QuAM pulse conditionally on a real-time Boolean Qiskit expression (`expr.Expr`), lowered directly to QUA `pulse.play(condition=...)` without a Qiskit `if_test` or QUA branching. Use `QuantumCircuit.conditional_play(pulse_name, condition, qubit)` or the standalone `add_conditional_play(qc, pulse_name, condition, qubit)`, after registering the pulse(s) on the backend with `backend.register_conditional_play(*pulse_names)` (several pulses in one call are resolved all-or-nothing and synchronized once). Each play is wrapped in a Qiskit `box` so that the condition's classical dependencies are tracked by the DAG and cannot be reordered by `transpile()`; the OpenQASM 3 exporter rejects a conditional-play box whose contents were altered. Registering two pulse labels that map to the same operation name (e.g. `"-y90"` and `"minus_y90"`) emits a warning.
- **`y` and `sxdg` basis gates** — `add_basic_macros()` now also maps `y` to the `y180` pulse and `sxdg` to the `-x90` pulse (both standard in the flux-tunable transmon pipeline it targets).
- **`barrier` and `store` in the backend `Target`** — QM backend Targets now define both as global, variable-width operations (like control flow), so native-operation checks treat them as genuinely supported rather than special-casing them. Neither enters the qm-qasm operations mapping: qm-qasm lowers `store` to a QUA assignment and `barrier` to a QUA `align`.
- **`missing_operation_names(qc, target)`** (`qiskit_qm_provider.utils.circuit.target_support`) — returns the operation names in a circuit, including those nested in control flow, that a `Target` does not define; used by `ensure_native_operations` and the private conditional-play registration check.

### Changed

- **Backend helpers live under `qiskit_qm_provider.utils`** — circuit checks, measurement helpers, QUA-script dumps, calibration keys, gate-name aliases, QuAM validation, and Quarc struct assignment each have their own submodule. `get_measurement_outcomes`, `assign_struct_with_table`, `get_qua_script`, `dump_qua_script`, and `pack_register_to_int` are still exported from `qiskit_qm_provider` and `qiskit_qm_provider.backend`. Unregistered conditional-play pulses are still reported by `ensure_native_operations`; the check itself is the private `_raise_for_unregistered_conditional_plays` in `qiskit_qm_provider.conditional_play`.
- **Measurement-outcome docs** — user guides and the backend API index describe only `comp.outputs` on `QuaCircuitCompilation`; the dict helper `get_measurement_outcomes` is no longer documented as a parallel path.
- **`add_basic_macros` import from `backend.backend_utils` is deprecated** — import it from `qiskit_qm_provider` or `qiskit_qm_provider.quam_macros.superconducting`. That old path warns and still returns the function; other names that used to live on `backend_utils` are not re-exported.
- **Circuit inputs are bound by Qiskit object, with clear errors** — `quantum_circuit_to_qua` now reads the `input` declarations the exporter actually emitted (`QMOpenQASM3Exporter.export` returns them) and binds each one to its table field or dictionary entry by Qiskit object, instead of predicting identifiers. Inputs the exporter renames now compile, on stock Qiskit as well as on the branch: a parameter whose name is not an identifier (`theta.1`, exported as `_theta_1`), and, on the branch, a vector named like a gate or keyword (a vector `x` next to the `x` gate — e.g. `zz_feature_map`'s — exported as `x_0`; also `rx`, `U`, `input`). A missing binding, or one of the wrong shape (scalar vs. array, array length), raises `ValueError` naming each input; qm-qasm otherwise failed with an opaque `Qua1GenerationPass failed`, or silently broadcast a scalar into every element of an array input. Fields that match no input are still ignored, so a shared table can be passed whole.
- **Array binding consistency** — sampler/estimator PUBs bind vector values by parameter identity for compile-time assignment, local streaming and IQCC hooks, zero-filling unused vector entries. Ambiguous input names are rejected, string dictionary keys such as `"t[2]"` or `"t"` follow the same rules as the objects, and singleton array row copies now work.
- **Dense parameter rows for primitives** — with `INPUT_STREAM`/`IO1`/`IO2`, each binding now travels as one raw row of `len(circuit.parameters)` floats (one push and one advance per binding instead of one per parameter), scattered in QUA into the circuit's fields; only used entries are assigned. On stock Qiskit, vector fields are created only for elements the circuit uses (matching the exporter).
- **Execution plans for both primitives** — `QMSamplerJob` and `QMEstimatorJob` build one plan per PUB (`SamplerPlan` / `EstimatorPlan` in `qiskit_qm_provider.job.primitive_plan`, sharing a `PrimitivePlan` base), exposed on `job.runtime_pubs`. The QUA builders (`sampler_program`, `estimator_program`, `plan_*_programs`) now take plans. The private `_ExecutionPlan` and the unused `observables_to_indices` are removed. `EstimatorPlan.from_pub` requires the estimator readout that `QMEstimatorV2.validate_estimator_pubs` adds (a `__c` register and one `obs_j` basis input per measured qubit) and raises `ValueError` without it and for observables acting on a qubit outside `__c`.
- **Packed estimator observables** — per-qubit basis codes are packed 2 bits per qubit (15 qubits per word) into one int row per binding, `[n_groups, words...]`, unpacked in QUA with shifts and masks. This replaces one push per qubit per observable plus the shared `obs_length_var`; IO1/IO2 exchange only the used part of a row, and a row shared by every binding is sent once. `QMEstimatorOptions(input_type=OPNIC)` (or setting it before `run()`) now raises `NotImplementedError` up front.
- **Single sync-hook generator** — `post_hook_sampler.generate_sync_hook_sampler` and `post_hook_estimator.generate_sync_hook_estimator` are replaced by `qiskit_qm_provider.job.sync_hook.generate_sync_hook(plans)`, shared by both primitives. Each push target is listed once (`SPECS`) and every push refers to it by index, so the hook grows by one short entry per binding.
- **Estimator measured qubits** — circuits and observables are taken as ISA, and the estimator measures exactly the qubits on which some observable of the PUB acts non-trivially (the union over the observables array), including qubits no gate touches and ancillas, instead of every qubit carrying a gate. New helper `job.primitive_plan.estimator_measured_qubits(observables)`, returning qubit indices.
- **Identity terms are evaluated exactly** — `<I> = 1` with zero variance, so identity terms are no longer measured: they contribute their coefficient in post-processing. A PUB whose terms are all identities is not compiled, sent or run (Qiskit's `BackendEstimatorV2` measures one qubit for it instead).
- **Breaking: estimator readout names reserved under `__`** — the per-qubit basis inputs added by `QMEstimatorV2.validate_estimator_pubs` are now `__obs_j` (were `obs_j`), next to the `__c` readout register, and a PUB circuit with a classical register, variable or stretch whose name starts with `__` is rejected with a `ValueError` naming them (previously only `__c` was checked, and a user input named `obs_0` failed with Qiskit's name-conflict error). Circuits prepared by hand for `EstimatorPlan.from_pub` must use the new names.
- **Final measurements are kept, as in `BackendEstimatorV2`** — `QMEstimatorV2.validate_estimator_pubs` no longer removes a PUB circuit's final measurements; it only appends the `__c` readout, like Qiskit's `BackendEstimatorV2`. Pass estimator circuits without final measurements (`qc.remove_final_measurements(inplace=False)`): a final `measure_all` now collapses the state before the basis rotations. This also removes the "Trying to add QuantumRegister to a QuantumCircuit having a layout" warning that `remove_final_measurements` raised on every ISA PUB.
- **Non-Pauli observables rejected clearly** — `SparseObservable` projector terms (`0`, `1`, `+`, `-`, `r`, `l`) now raise a `ValueError` pointing to `SparseObservable.as_paulis()`, instead of an opaque `QiskitError` from `PauliList`.
- **Parameter values must fit QUA `fixed`** — `QMSamplerV2` and `QMEstimatorV2` now reject, at job creation, any PUB with a bound parameter value outside the 4.28 fixed-point range [-8, 8) or not finite (`ValueError` naming the PUB, parameter and binding). Such values were previously sent or preloaded unchecked and could not be represented on the controller. The primitives guide now warns that expressions over parameters are evaluated in real-time fixed-point arithmetic and can still overflow, and points to QUA's `Cast` utilities for rescaled kernels.
- **`Parameter.push_io_values`** — the IO pause/set/resume loop now lives in one place; `push_to_opx` and the estimator's partial observable rows both use it.
- **`job/_pub_parameters.py`** renamed to `job/pub_parameters.py` (internal; not exported).
- **`ParameterVector` handling in `ParameterTable.from_qiskit`** — when the installed Qiskit has the `PARAMETER_VECTORS_AS_ARRAYS` exporter feature (the `full_real_time_expr` branch), a `ParameterVector` becomes **one** array `Parameter` named after the vector, of the vector's full length; the full vector is declared whenever the `filter_function` keeps any of its used elements. That field binds either export of the vector: the full-length `input array[float[64], N]` (`export_vector_as_array=True`) or one scalar input per used element (option off), whose inputs are bound to the array's cells. Stock Qiskit keeps per-element `_name_i_` fields. Dictionary bindings to `quantum_circuit_to_qua` take a whole vector (or its name) in both exports; a single-element key binds only an element exported as a scalar input.
- **Breaking (stock Qiskit): `ParameterTable.from_qiskit` creates fields only for used vector elements** — matching what the exporter declares. Previously every element of a `ParameterVector` got a `_name_i_` field even if the circuit never used it, so code that pushes or assigns values for unused elements by name (e.g. `table.push_to_opx({"_theta_3_": ...})`) now fails on the missing field; drop those entries.
- **Breaking: `quantum_circuit_to_qua()` only accepts circuits transpiled for the backend** — the OpenQASM 3 lowering needs circuits expressed in native operations on physical qubits, so `quantum_circuit_to_qua()` (and everything built on it, such as `backend.run()` and the primitives) now raises `ValueError` unless the circuit:
  - has been laid out on the backend's physical qubits: it carries a `TranspileLayout` and a single canonical `q` register of exactly `backend.num_qubits` qubits (`ensure_circuit_physical` in `qiskit_qm_provider.utils.circuit.physical_layout`);
  - uses only operations supported by `backend.target` on the qubits they act on, as checked by Qiskit's `GatesInBasis` pass (`ensure_native_operations` in `qiskit_qm_provider.utils.circuit.target_support`). Errors name what is wrong: unregistered conditional-play pulses, then operation names the Target does not define, then the first operation used on unsupported qubits or parameters.

  Circuits that previously compiled without transpilation (hand-built `QuantumCircuit(n)`, or `transpile(..., basis_gates=...)` without the backend) must now be transpiled against the backend first. Both checks only inspect the circuit and never modify it.
- **Barriers now compile to QUA `align`** — `quantum_circuit_to_qua()` no longer strips `barrier` statements from the exported OpenQASM 3; qm-qasm lowers each one to an `align` over every element of the barrier's qubits. Circuits containing barriers (including those `transpile` or `measure_all` insert) now synchronize those qubits at that point.
- **Minimum Qiskit version is now 2.2** — the OpenQASM 3 exporter relies on `qiskit.qasm3.DefcalInstruction`/implicit defcals and the physical-circuit form introduced in Qiskit 2.2.
- **TLS-secured QM SaaS connection** — `QmSaasProvider.get_backend()` now builds its `QuantumMachinesManager` with `QuantumMachinesManager(**instance.qmm_connection_params)`, which adds the TLS credentials required by the QM cloud simulator from QOP 3.8.1 onwards (connections without them fail after the October 8th, 2026 server update). Replaces the deprecated `host`/`port`/`default_connection_headers` form.
- **`qm-saas` extra** now requires `qm-saas>=1.2.0,<2` (first release exposing `qmm_connection_params`).

### Fixed

- **Chunked jobs with streamed inputs no longer deadlock on hardware** — when `max_circuits` splits a sampler or estimator job into several queued programs, each chunk is now waited on and fed its inputs before the next one: previously every chunk had to be running before any input was pushed, but a queued program only starts once the previous one has finished, so the first chunk waited forever for its inputs (`INPUT_STREAM`, `IO1`, `IO2`; compile-time inputs and IQCC were not affected). `await_running_jobs` takes an `on_running(job, chunk)` callback for this.
- **Identity-only estimator jobs** — when every term of every PUB is the identity, the job sends nothing to the QM (it previously still opened a connection), and an IQCC job returns its result instead of raising `RuntimeError: IQCC job has not been submitted yet`.
- **Scalar named like a vector element** — with vector arrays, a scalar input named `_t_2_` next to `ParameterVector` `t` is now rejected as ambiguous; when `t[2]` was exported as a scalar input it silently bound to that scalar instead of the vector's array cell.
- **PUBs without parameters or without bindings** — a PUB whose circuit has no parameters gets no parameter table and no binding loop: the sampler enters the shot loop directly (running `k` times the shots, in binding order, for a `(k, 0)` bindings array, which previously returned the wrong number of shots), and the estimator loops only over its observable groups. A PUB with zero bindings (e.g. `np.zeros((0, n))`) is valid: it is not compiled, sent or run, and its result is empty with the PUB's shape (`(0,)`), as with Qiskit's reference primitives. It previously failed with an opaque `QUA2DArray` error (sampler) or a `ValueError` (estimator); a job whose PUBs are all such sends nothing to the QM.
- **Deprecated `CircuitInstruction` unpacking** — `logically_active_qubits` no longer iterates circuit instructions as tuples, which Qiskit deprecated in 1.2 and removes in 3.0 (every estimator PUB emitted the `DeprecationWarning`).
- **Out-of-range parameter error on NumPy 2** — the message of the `[-8, 8)` fixed-point check printed `np.float64(12.5)` instead of `12.5`.
- **Estimator expectation values on multi-qubit observables** — three ordering errors combined to give wrong results whenever an observable acted on more than one qubit: `get_non_trivial_observables` reversed the qubit order of the basis rotations, `counts_from_estimator_stream` read `__c` big-endian, and measured Paulis were compacted with Qiskit's per-basis measurement indices although `__c` holds every measured qubit. Grouping now happens once per distinct set of Pauli terms (shared by the bindings that request it), so metadata, basis codes and the `__c` stream share one order (`abelian_grouping=False` no longer misreads or crashes).
- **Estimator result shape** — parameter indices are broadcast against the observables before post-processing, as in Qiskit; PUBs with more observables than bindings (e.g. several observables, no parameters) no longer fail in `result()`.
- **Estimator run options** — `simulate` and `compiler_options` nested under the estimator's `run_options` now reach program submission.
- **IQCC estimator sync hook** — attached whenever a PUB streams inputs, including PUBs that stream observables but have no circuit parameters.
- **Estimator table names** — tables are named by PUB index, fixing duplicate pool registration when two PUBs share a circuit name.
- **`QMBackend.qubit_mapping` element names** — the base backend mapped each qubit to its `channels` dict keys (`"xy"`, `"z"`, ...) instead of the channels' QUA element names (`"q0.xy"`, ...), so every `align` qm-qasm emitted targeted non-existent elements. `FluxTunableTransmonBackend` already overrode this correctly.

## [0.3.5] - 2026-09-23

### Added

- **`GPIGate` / `GPI2Gate`** — IonQ-native single-qubit gates (matching Qibo's `GPI`/`GPI2` convention) added to `additional_gates.py` alongside the existing `SY`/`SYdg`/`CR`/`FSim` gates, with `QuantumCircuit.gpi()` / `QuantumCircuit.gpi2()` convenience methods and `"gpi"` / `"gpi2"` entries in `get_extended_gate_name_mapping()`.

### Changed

- **`look_for_standard_op` gate-name aliases** — `"rx90"` now maps to `"sx"`, alongside the existing `"x90"` / `"x/2"` aliases.
- **`QuantumCircuit.sy()` / `sydg()` / `cr()` / `fsim()` / `gpi()` / `gpi2()`** — all now accept an optional `label` argument, forwarded to the underlying gate's constructor.
- **Optional Qiskit Pulse** — `QISKIT_PULSE_AVAILABLE` is probed with `importlib.util.find_spec` and no longer emits an `ImportWarning` on Qiskit 2.x (where Pulse was removed). Pulse classes are imported only inside Pulse-gated methods; `QiskitChannel` is a type hint (`Any` at runtime) for `channel_mapping`. Circuit-only import and `backend.run()` paths no longer load `qiskit.pulse`. `requires_qiskit_pulse` is unchanged.

### Fixed

- **Value-hashable `operation_key`** — `qm_qasm.OperationIdentifier` has no `__eq__`/`__hash__`, so using it directly as a `dict`/`set` key treated logically identical operations as distinct and silently duplicated them. Introduced `operation_key(name, number_of_params, qubits)` — a plain, value-equal tuple — used across backend operation mappings and `has_conflicting_calibrations` for correct deduplication/overwriting. Real `OperationIdentifier` instances are now built only once, when constructing the `Compiler`/`HardwareConfig`. No public API changes.

## [0.3.4] - 2026-08-13

### Changed

- **Canonical IQCC cloud path** — `backend.run()` on IQCC backends always uses `CloudQuantumMachine.execute` via `QMJob` / `CloudQMJob`. Removed the undocumented legacy `IQCCJob` path that treated `IQCC_Cloud` itself as the quantum machine. Prefer `IQCCProvider.get_backend()` so QuAM `connect()` yields `CloudQuantumMachinesManager`.
- **Unified QM program submission** — cloud and simulation always use ``qm.execute`` (simulation via ``simulate=SimulationConfig``); real hardware enqueues via OPX1000 ``add_to_queue`` with OPX+ ``queue.add`` fallback, replacing ``compile`` + ``queue.add_compiled``. Shared helpers live in ``qm_execution_options``.
- **`quam` lower bound** raised to `>=0.4.2` so the provider stays importable with current `qm-qua` (0.4.1 imported a removed `AmpValuesType`).

### Fixed

- **`QMJob.submit()` cloud execution** — trim `compiler_options` (and route through `CloudQuantumMachine.execute(options=…)`) when the backend uses `CloudQuantumMachinesManager`, matching the IQCC cloud client API and avoiding `TypeError` on every cloud job submission.

## [0.3.3] - 2026-08-07

### Fixed

- **`add_basic_macros` default CZ gate** — pass `flux_pulse_qubit=` (not the obsolete `flux_pulse_control=`) when constructing quam-builder’s `CZGate`, so default two-qubit macros seed correctly instead of failing silently behind the existing warning.

## [0.3.2] - 2026-07-22

### Changed

- **`add_basic_macros` relocated** to `qiskit_qm_provider.quam_macros.superconducting` (re-exported from `qiskit_qm_provider`). Prefers single-qubit macros from `quam-builder`, with a fallback to the provider’s local copies, and accepts `**reset_macro_kwargs` for `ResetMacro`. Importing it from `qiskit_qm_provider.backend.backend_utils` is deprecated.

### Fixed

- **`QUAArray.push_to_opx` / `QUA2DArray.push_to_opx`** — signatures now match the base `Parameter` API: `job` may be a `RunningQmJob` or `JobApi` (optional). The optional `qm` argument is retained only for legacy back-compat; current QUA drives IO through the job interface.

## [0.3.1] - 2026-07-02

### Fixed

- **Duplicate `INPUT_STREAM` declarations** — `Parameter.declare()` reuses an existing same-named stream in the current QUA program scope when type/length match (emits `UserWarning`), instead of raising “Variable already declared”. Common with chunked estimator/sampler plans that share streamed field names. Incompatible collisions still raise `ValueError`.
- **`estimator_program` `obs_length_var`** — declared once at program scope and only `rcv()`’d per plan, avoiding per-plan re-declaration in multi-plan programs.

## [0.3.0] - 2026-07-02

### Added

#### Chunked execution (`max_circuits_spec`)

- **`max_circuits` backend option** — `QMBackend`, `QMSamplerV2`, and `QMEstimatorV2` accept a `max_circuits` option.  When the circuit/PUB count exceeds it, execution is automatically split into consecutive QUA programs; results are stitched back transparently.
- **`compute_chunk_layout()`** in `qua_programs.py` — plans the per-program circuit/PUB grouping for a given `max_circuits` limit.
- **`compute_locator()`** in `qua_programs.py` — inverts a chunk layout into a `global_index → (chunk_idx, local_idx)` mapping used during result assembly.  Previously inlined as a dict comprehension in three separate places.
- **`stream_assembly.py`** — shared helpers `bit_array_from_stream()` and `bit_array_from_measurement_stream()` that assemble `BitArray` results from QUA streams, handling both standard QM (flat output) and IQCC cloud (may carry an extra leading dimension) via a `reshape` normalisation step.
- **`IQCCJobMixin.status()`** — IQCC-aware status check: since `CloudJob.status` is unconditionally `"completed"` (IQCC execution is synchronous), this method inspects `_run_data["stderr"]` instead.  Returns `JobStatus.ERROR` when the cloud stderr contains a Python traceback, `JobStatus.DONE` otherwise.  Inherited by both `IQCCSamplerJob` and `IQCCEstimatorJob`.
- **`aggregate_job_statuses()`** in `iqcc_job_mixin.py` — shared status-aggregation helper (worst-case across all chunk jobs) used by both `QMJob.status()` and `QMPrimitiveJob.status()`.
- **`QMJob.program`** — backward-compatible property alias for `programs[0]`.

### Fixed

#### Chunked execution (`max_circuits_spec`)

- **`IQCCJob.submit()`** — was executing only `programs[0]`, silently dropping all chunks beyond the first for IQCC `backend.run()` calls with chunking; now loops over all programs.
- **`QMSamplerJob.submit()` simulate path** — was always submitting only `programs[0]`; now iterates all chunks so multi-program simulation works correctly.
- **IO1/IO2 parameter push on `QmPendingJob`** — `wait_until_job_is_paused` requires a `RunningQmJob`; the compile+add_compiled refactor passed a pending job. Both `QMSamplerJob.submit()` and `QMEstimatorJob.submit()` now call `pending.wait_for_execution()` before pushing parameters and store the resulting `RunningQmJob` in `_qm_jobs`.
- **`Result(results=…)` single-circuit** — was passing a bare `ExperimentResult` (not a list) for single-circuit runs, causing `TypeError` on `result.results[0]`; always passes a list now.
- **IQCC partial execution** — `IQCCSamplerJob` and `IQCCEstimatorJob` now initialize `self._qm_jobs = []` before the execute loop and append incrementally; a mid-loop failure no longer leaves `_qm_jobs = None`, making partial state visible.
- **`IQCCJobMixin.status()` / `result()` empty-list false DONE** — `self._qm_jobs = []` (set before the loop, never appended to when the first `execute()` raises) previously passed the `is None` guard and returned `JobStatus.DONE`; guard now uses a falsy check covering both `None` and `[]`.
- **`wait_for_execution()` mid-loop failure** — a failure on chunk *k* previously left a partial `_qm_jobs` list and caused a cryptic `IndexError` in `_result_function`; now raises `RuntimeError` naming the chunk index and the circuit/PUB indices in that chunk so the user knows which job to isolate.
- **`QMJob.submit()` dead cloud kwargs** — removed a `CloudQuantumMachine` kwargs block (timeout / terminal_output) that silently dropped the user-specified timeout on multi-program cloud jobs; `IQCCJob` overrides `submit()` entirely so the path was unreachable.
- **Status mapping duplication** — `QMJob.status()` and `QMPrimitiveJob.status()` had identical status-string → `JobStatus` mapping dicts inlined separately; both now delegate to the shared `aggregate_job_statuses()` helper.

#### Quarc hybrid integration (`quarc-support`)

- **Quarc integration for OPNIC stream assignment** — `ParameterPool.prepare_opnic_quarc_hybrid_packets()` deterministically attaches incoming/outgoing stream IDs to `ParameterTable` and standalone `Parameter` objects before QUA variable declaration, replacing the legacy `opnic_wrapper`-based path.
- **`default_quarc_struct_name()`** — helper to keep struct naming consistent between this provider and the Quarc build system.
- **`quarc_emit.py`** — translation helpers from QUA types to Quarc annotations (`quarc_atomic_for`, `quarc_annotation_for`, `quarc_direction_for`, `build_quarc_struct`) without prematurely consuming global stream IDs.
- **Dual Quarc emission pipelines** in `ParameterPool`:
  - **Module-first** — `ParameterPool.from_quarc_module(module)` rebuilds OPNIC-backed `ParameterTable` / `Parameter` objects from an existing Quarc `BaseModule`.
  - **Parameter-first** — `ParameterPool.to_quarc_module()` lazily binds (or creates) a Quarc module and defers struct emission until explicitly requested.
- **Optional `[quarc]` extra** in `pyproject.toml` for installing Quarc as a peer dependency.
- **Lazy Quarc import** — top-level `import qiskit_qm_provider` no longer requires `quarc`; `QiskitQMModule` is loaded via `__getattr__` and `QUARC_AVAILABLE` probes availability at import time.

#### OPNIC host↔OPX transport (formerly DGX_Q)

- **`InputType.OPNIC`** — rebranded from `InputType.DGX_Q` for classical host packet communication over the OP Network Interface Card.
- **Quarc-managed struct handles** — `ParameterTable` and `Parameter` delegate `push_to_opx`, `fetch_from_opx`, and `stream_back` to Quarc `QuaStructHandle` endpoints instead of manual packet assembly.
- **Standalone OPNIC parameters** — synthetic `ParameterTable` adapter lets OPNIC `Parameter` instances perform I/O without explicit table binding; lazy promotion on first transport call.
- **Bidirectional stream resolution** — `incoming_stream_id` and `outgoing_stream_id` properties on `Parameter` and `ParameterTable`, resolved from Quarc handle metadata.
- **`Parameter.assign_from_io(io_index)`** — non-blocking assignment of global IO registers (`IO1` / `IO2`) to scalar variables without pausing QUA execution.

#### QiskitQMModule

- **`QiskitQMModule`** — `quarc.BaseModule` extension unifying OPNIC and non-OPNIC parameter declaration, persistence, and reconstruction:
  - Automatic registration with `ParameterPool` at construction.
  - Sweeps pre-existing OPNIC tables and pending standalone parameters onto the module.
  - `parameter_specs` field for serialising non-OPNIC parameters.
  - `reconstruct_non_opnic()` and `iter_all_params()` for round-trip module state.
- **`QiskitQMModule.bind_connection(qmm)`** — serialises QMM host/port into module JSON; invoked automatically on first `QMBackend.qm` access.
- **`QiskitQMModule.get_running_job()`** — retrieves the active QM job in both local backend mode and IQCC sync-hook mode.
- **Default module allocation** — `ParameterPool.quarc_module()` / `to_quarc_module()` lazily allocate a `QiskitQMModule` (instead of a plain `quarc.BaseModule`) and sweep pending parameters on first access.

#### Parameter and ParameterTable API

- **Unified interface** — `Parameter` and `ParameterTable` share canonical method names (`declare`, `rcv`, `declare_stream`, `reset_qua`) so callers never branch on concrete type.
- **1-field struct promotion** — `ParameterPool.from_quarc_module()` promotes Quarc structs with exactly one field to a standalone `Parameter`.
- **`QuaFieldTable` mixin** — shared field-access pattern for table-like QUA objects.
- **QUA program scope guards** — `is_inside_scope()`, `require_qua_program()`, and `@requires_qua_program` enforce that QUA variable accessors are used inside `with program():`.
- **`assign_struct_with_table()`** — QUA macro copying declared `ParameterTable` values into a Quarc `QuaStructHandle` with runtime validation and shape checks.
- **`pack_register_to_int()`** — helper for bit-packing classical register values into a QUA integer scalar.
- **Zero-argument `push_to_opx()`** — `Parameter.push_to_opx()` and `ParameterTable.push_to_opx()` default to streaming current values when called without arguments.

#### Circuit compilation and measurement outputs

- **`QuaCircuitCompilation`** — first-class wrapper for qm-qasm compilation results returned by `QMBackend.quantum_circuit_to_qua()`.
- **`MeasurementOutcomeTable` (`comp.outputs`)** — compilation-local measurement handles wired from qm-qasm `result_program`:
  - `comp.outputs["c"]` / `comp.outputs.get_parameter("c")` for bool vars and field handles.
  - `comp.outputs.state_ints["c"]` for lazy-packed integer scalars.
  - `comp.outputs.streams["c"]` for per-field `declare_stream()` handles.
- **`MeasurementRegisterField`** — per-register handle exposing measurement outcome accessors without polluting the runtime `ParameterPool`.
- **Loose QASM3 classical bits** — `_LooseBitRegister` and extended `get_measurement_outcomes()` surface unassigned classical bits under synthetic `_bitN` keys.

#### Gates and backend utilities

- **`FSimGate`** — parameterized fSim gate (θ, φ) with `QuantumCircuit.fsim()` monkey-patch.
- **Expanded additional gates** — `SYGate`, `SYdgGate`, and `CRGate` retained and registered in `get_extended_gate_name_mapping()`.
- **`add_basic_macros(backend)`** — exported utility to populate backends with standard single-qubit gate, measure, and reset macros; syncs the backend target automatically.
- **`quam_macros` module** — dedicated home for superconducting qubit macros (`MeasureMacro`, `ResetMacro`, `VirtualZMacro`, `DelayMacro`, `IdMacro`), extracted from `iqcc_calibration_tools`.
- **Non-destructive macro injection** — `add_basic_macros` only fills empty `qubit.macros` dicts and absent pair-level `cz` entries, preserving user-defined macros.
- **Improved flux-tunable channel mapping** — `FluxTunableTransmonBackend` qubit→element mapping returns variable-length channel tuples including TWPA pump channels when present.
- **`QMInstructionProperties` `quam_macro` support** — explicit QuAM macro input takes priority over `qua_pulse_macro` for property inference.

#### Runtime and cross-environment job access

- **`qiskit_qm_provider.runtime`** — helpers for hybrid classical/QUA workflows:
  - Sync-hook job reconstruction from IQCC CLI args (`-j/-q/-i/-p`).
  - Running-job polling compatible with QOP 3.x (`get_jobs(status=["Running"])`) and QOP 2.x (`list_open_qms()` fallback).

#### Self-contained IQCC sync hooks

- **Dependency-free sync-hook generation** — `generate_sync_hook_sampler()` and `generate_sync_hook_estimator()` now emit a sync hook that runs on **any** IQCC user's cloud runtime: the generated code imports only `from iqcc_cloud_client.runtime import get_qm_job` and no longer requires `qiskit_qm_provider` or `numpy` to be installed on the sync-hook side. Parameter tables and observable indices are serialised to plain-Python data; the push logic is rendered from Jinja templates (`qiskit_qm_provider/job/templates/`).
- **`INPUT_STREAM` and `IO1`/`IO2` over IQCC** — both delivery mechanisms are fully runnable from the sync hook. `INPUT_STREAM` pushes via `job.push_to_input_stream(name, value)`; `IO1`/`IO2` set `job.set_io_values(...)` with an inlined pause/resume loop (an in-template equivalent of `qualang_tools.results.wait_until_job_is_paused`), so `qualang_tools` is not required in the cloud runtime either.
- **Per-parameter type coercion** — streamed values are coerced according to each parameter's QUA type (`int` / `fixed` / `bool`) instead of a blanket `float()` cast.
- **`jinja2`** added to the `[iqcc]` optional extra.

#### Documentation

- **[Primitives guide — Running on IQCC Cloud](docs/primitives.md)** — explains the self-contained sync-hook behavior, supported `input_type` mechanisms (`INPUT_STREAM`, `IO1`/`IO2`), and the `OPNIC` limitation on IQCC jobs.
- **[Measurement outputs guide](docs/measurement_outputs.md)** — locality model, accessor contract, RL reward and QEC worked examples.
- **[Error-correction workflow guide](docs/error_correction.md)** — expanded hybrid QEC patterns (syndrome streaming, detection events, decoder integration).
- **Parameter table and backend API docs** — updated Sphinx stubs, scope-guard reference, and OPNIC workflow documentation.

### Changed

- **IQCC sync-hook generation rewritten** — the previously emitted hook re-declared `ParameterTable`/`Parameter` objects and called `push_to_opx()` (requiring `qiskit_qm_provider` + `numpy` in the cloud runtime). It now injects plain data into a Jinja template that talks to the QM job directly. `QMSamplerV2.run()` / `QMEstimatorV2.run()` raise `NotImplementedError` when `input_type=InputType.OPNIC` is combined with an IQCC job (`IQCCSamplerJob` / `IQCCEstimatorJob`), since OPNIC transport is not yet available over the cloud sync hook — use `INPUT_STREAM` or `IO1`/`IO2`.
- **`QMBackend.quantum_circuit_to_qua()`** now returns `QuaCircuitCompilation` instead of a raw qm-qasm `CompilationResult`.
- **`InputType.DGX_Q` renamed to `InputType.OPNIC`** throughout the codebase, primitives options, and documentation. OPNIC is presented as a QUARC-backed host↔OPX transport; the legacy "DGX Quantum" naming is fully retired from code and docs.
- **`Parameter.dgx_struct` / `opnic_struct` replaced by `Parameter.struct_type`** (and equivalent on `ParameterTable`); the `opnic_struct` name was removed outright (see Removed).
- **`ParameterPool.prepare_dgx_quarc_hybrid_packets()` renamed to `prepare_opnic_quarc_hybrid_packets()`.**
- **Internal callers migrated to canonical names** — library code (`qua_programs`, `qm_backend`, `pulse_support_utils`) now calls `declare()` / `rcv()` instead of the deprecated `declare_variable(s)` / `load_input_value(s)`, so building sampler/estimator programs no longer self-emits `DeprecationWarning`.
- **`get_measurement_outcomes()`** sources every entry from `comp.outputs` and exposes `is_array` / `length` keys (`Parameter` convention). Loose clbits are returned under per-bit keys `_bit0`, `_bit1`, … (one entry each) instead of a single packed `_bit` register — loose bits are independent and are no longer packed into one integer. `state_int` is always lazily available (no construction-time gate).
- **`MeasurementRegisterField`** now follows the `Parameter` shape convention — it carries `length` (`0` for a scalar loose clbit, the register width otherwise) and exposes `is_array` (`length > 0`) and `length` as part of the shared `TableFieldProtocol`. Dropped the bespoke `register_size` / `var_is_array` state; `pack_register_to_int` remains the single place that adapts packing to the wired QUA variable.
- **`MeasurementRegisterField` / `MeasurementOutcomeTable.from_compilation` / `QuaCircuitCompilation`** dropped the `compute_state_int` constructor argument — `state_int` is a lazy property computed only on access.
- **`Parameter` INPUT_STREAM declaration** uses the current qm-qua API (`declare_input_stream(type, name=..., value=...)`); output streams still use `declare_stream()`. A QUA 1.3 client/`stream_id` path is scaffolded in `parameter.py` but remains commented until qm-qua ≥ 1.3 (note: `declare_output_stream` is not exported in qm-qua 1.2.x).
- **`ParameterTable.fetch_from_opx()`** uses bulk `recv(size, index)` instead of per-element iteration.
- **`QMBackend.compile()` `param_table` typing** broadened from `List[...]` to `Sequence[...]`.
- **`IQCCProvider`** always defaults to `FluxTunableQuam` when `quam_cls` is not supplied (removed conditional fallback import).
- **`add_basic_macros`** stores pulse references by name (string) in QuAM macro definitions.
- **Deprecation aliases** — historical method names (`declare_variable(s)`, `load_input_value(s)`, `reset_var(s)`, `declare_streams`) preserved via `_DeprecatedAlias` descriptor; access emits `DeprecationWarning` (scheduled removal in v1.2).

### Deprecated

- **`declare_variables` / `declare_variable`** → use `declare`.
- **`load_input_values` / `load_input_value`** → use `rcv`.
- **`reset_vars` / `reset_var`** → use `reset_qua`.
- **`declare_streams`** → use `declare_stream`.

### Removed

- Legacy **`opnic_wrapper`-based** OPNIC transport inside `ParameterPool`.
- **`OpnicPacketBinding` protocol** and internal OPNIC orchestration modules (`quarc_naming`, `quarc_live_module`).
- **`ParameterPool` manual stream-ID rebinding** methods (`rebind_parameter_table_id`, etc.) in favor of Quarc-managed handles.
- **`@opnic_check` decorator** (relaxed via standalone OPNIC adapter pattern).
- **`opnic_struct` alias** on `Parameter` / `ParameterTable` — removed outright (no deprecation cycle); OPNIC transport never shipped working, so it had no usable callers. Use `struct_type` for the Quarc struct type and `var` for the bound handle.
- **`_LooseBitRegister`** helper class and the monotonic output-table name counter (`reset_output_table_name_registry`) — table names are now derived from currently-live tables, so they no longer grow unboundedly across a long-lived process.
- Dependency on **`iqcc_calibration_tools`** for single-qubit macros (replaced by `quam_macros`).

### Fixed

- **Input stream declarations** — aligned with the installed qm-qua `declare_input_stream` signature (legacy `(type, name=..., value=...)` path active; QUA 1.3 client-target syntax prepared but not enabled).
- **ResetMacro pi_12 pulse forwarding** — corrected argument pass-through in single-qubit reset macro.
- **Quarc packet naming collision** — synthetic table names avoid matching single-parameter field names that trigger Quarc complaints.
- **Ill-defined Quarc `recv()` bypass** — workaround for incomplete Quarc receive API.
- **QUA array constructors** — `QUAArray` and `QUA2DArray` override `__new__` to prevent `qua_type` double-binding during positional init.
- **Backend utils docstring** cleanup in `add_basic_macros`.
- **QuAM macros folder location** corrected after module extraction.
