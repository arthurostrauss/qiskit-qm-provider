# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Changes on the `quarc-support` branch relative to `main` (22 commits, ~8.5k lines added).

### Added

#### Quarc hybrid integration

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

#### Documentation

- **[Measurement outputs guide](docs/measurement_outputs.md)** — locality model, accessor contract, RL reward and QEC worked examples.
- **[Error-correction workflow guide](docs/error_correction.md)** — expanded hybrid QEC patterns (syndrome streaming, detection events, decoder integration).
- **Parameter table and backend API docs** — updated Sphinx stubs, scope-guard reference, and OPNIC workflow documentation.

### Changed

- **`QMBackend.quantum_circuit_to_qua()`** now returns `QuaCircuitCompilation` instead of a raw qm-qasm `CompilationResult`.
- **`InputType.DGX_Q` renamed to `InputType.OPNIC`** throughout the codebase, primitives options, and documentation.
- **`Parameter.dgx_struct` renamed to `Parameter.opnic_struct`** (and equivalent on `ParameterTable`).
- **`ParameterPool.prepare_dgx_quarc_hybrid_packets()` renamed to `prepare_opnic_quarc_hybrid_packets()`.**
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
- Dependency on **`iqcc_calibration_tools`** for single-qubit macros (replaced by `quam_macros`).

### Fixed

- **Input stream declarations** — aligned with the installed qm-qua `declare_input_stream` signature (legacy `(type, name=..., value=...)` path active; QUA 1.3 client-target syntax prepared but not enabled).
- **ResetMacro pi_12 pulse forwarding** — corrected argument pass-through in single-qubit reset macro.
- **Quarc packet naming collision** — synthetic table names avoid matching single-parameter field names that trigger Quarc complaints.
- **Ill-defined Quarc `recv()` bypass** — workaround for incomplete Quarc receive API.
- **QUA array constructors** — `QUAArray` and `QUA2DArray` override `__new__` to prevent `qua_type` double-binding during positional init.
- **Backend utils docstring** cleanup in `add_basic_macros`.
- **QuAM macros folder location** corrected after module extraction.
