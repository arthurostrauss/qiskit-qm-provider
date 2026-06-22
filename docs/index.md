# Qiskit QM Provider

Welcome to the Qiskit-QM-Provider documentation!

This project is a **toolbox for crossing abstraction layers** between high-level circuits written in [Qiskit](https://www.ibm.com/quantum/qiskit) and low-level QUA control programs on Quantum Machines' [Quantum Orchestration Platform](https://docs.quantum-machines.co/latest/) (QOP).

**Author:** Arthur Strauss — *Centre for Quantum Technologies, National University of Singapore, in collaboration with Quantum Machines Ltd.*  
**License:** Apache 2.0 (see the `LICENSE.md` file in the repository).

```{toctree}
:hidden:

workflows
providers
backend
measurement_outputs
primitives
parameter_table
error_correction
changelog
API Reference <apidocs/qm>
GitHub <https://github.com/arthurostrauss/qiskit-qm-provider>
```

## Documentation map

This site has two complementary layers:

- **Guides** (sidebar pages below) — purpose, architecture, and practical snippets: *why* and *how* to use each part of the toolbox.
- **[API Reference](apidocs/qm.rst)** — auto-generated signatures and docstrings from the Python source: the authoritative reference for every class and method.

Runnable scripts live in the [examples](https://github.com/arthurostrauss/qiskit-qm-provider/tree/main/examples) folder.

## What this toolbox gives you

The provider supports **two layers of use**:

1. **Traditional** — [`QMBackend.run()`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) and V2 primitives ([`QMSamplerV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerV2.rst), [`QMEstimatorV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMEstimatorV2.rst)) compile Qiskit circuits to QUA and execute them on QOP, with optional real-time parameter streaming.
2. **Extended (hybrid)** — embed Qiskit circuits as **QUA macros** inside larger programs via [`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst), with [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) as the classical-quantum contract.

Underneath both paths, the backend reads your [QuAM](https://qua-platform.github.io/quam/) machine description to build a Qiskit `Target` (connectivity, native gates, qubit properties), so the full transpiler stack applies. You can bring your own `QuamRoot` and [`QMBackend`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) subclasses, or start from [`FluxTunableTransmonBackend`](apidocs/stubs/qiskit_qm_provider.backend.FluxTunableTransmonBackend.rst).

## Installation

```bash
pip install qiskit-qm-provider
```

For IQCC cloud access and QM SaaS simulation:

```bash
pip install qiskit-qm-provider[iqcc]
pip install qiskit-qm-provider[qm_saas]
```

## Using Qiskit Experiments with this provider

Many users want to combine [Qiskit Experiments](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-experiments) with this provider (especially via [`IQCCProvider`](apidocs/stubs/qiskit_qm_provider.providers.iqcc_cloud_provider.IQCCProvider.rst)). Two caveats are worth reading **before** you start:

### 1. Match the tool to the execution model

Qiskit Experiments typically expresses characterization as a **large batch of pre-defined circuits** — often nearly identical except for one swept parameter (Rabi amplitudes, Ramsey delays, and so on). That pattern mirrors **AWG-style playback**: every variant is compiled and loaded ahead of time.

QUA is designed differently: one program with **real-time loops, streaming, and on-device parameter updates**. For many calibration sweeps, [Qualibrate](https://qualibrate-docs.quantum-machines.co/) and [qua-libs](https://github.com/qua-platform/qua-libs) are the natural QUA-native counterparts.

This provider shines when you **invert the usual assumption**: use Qiskit for circuit *authoring* (synthesis, visualization, transpilation, portability) and wrap those circuits as **QUA macros** inside richer real-time programs — not when you treat Qiskit as the *entire* control stack and never engage with QUA.

IQCC + Experiments (see `examples/iqcc_t1_experiment.py`) is supported and useful, but ask whether a batch-of-circuits experiment is really what you need, or whether a parameterized QUA loop (with [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) / [`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst)) would be more efficient on QOP. See [Workflows — Qiskit Experiments](workflows.md#qiskit-experiments-iqcc-with-caveats) for the full discussion.

### 2. Classified measurement outcomes only

Experiments that need raw **I & Q**, kerneled shots, or other non-discriminated data are **not yet supported** end-to-end. Only **classified** results (0/1 counts / bitstrings) are reliably returned today. Contributors are welcome if IQ/kernels matter for your workflow.

*(Note: `meas_level` options may appear in the API, but non-classified paths are not production-ready.)*

## Where to go next

Use the sidebar to browse guides, then dive into the [API Reference](apidocs/qm.rst) for signatures:

- **[Workflows](workflows.md)** — routing guide for the five main paths through the toolbox.
- **[Providers](providers.md)** — obtain a backend for local QOP, QM SaaS, or IQCC.
- **[Backend & Utilities](backend.md)** — [`QMBackend`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst), embedding, utilities, custom gates.
- **[Measurement outputs](measurement_outputs.md)** — [`QuaCircuitCompilation`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation.rst), [`MeasurementOutcomeTable`](apidocs/stubs/qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable.rst), [`MeasurementRegisterField`](apidocs/stubs/qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField.rst).
- **[Primitives](primitives.md)** — [`QMSamplerV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMSamplerV2.rst) and [`QMEstimatorV2`](apidocs/stubs/qiskit_qm_provider.primitives.QMEstimatorV2.rst).
- **[Parameter Table](parameter_table.md)** — [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) and real-time data flow.
- **[Error-Correction Workflow](error_correction.md)** — hybrid EC pattern with [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst).
- **[API Reference](apidocs/qm.rst)** — autodoc pages for all public classes and functions.

## Design philosophy

Working at the intersection of **algorithms** and **control** is hard. This provider is designed to:

- keep the **Qiskit user experience** for circuit design and experimentation,
- expose **QOP/QUA features** (real-time feedback, streaming, hybrid control) without rewriting everything as low-level QUA,
- make it easy to **embed Qiskit circuits inside larger QUA programs** with a shared calibration model.

The compiler's value is not "hide QUA behind Qiskit forever." It is **frictionless advanced QUA** — real-time logic, streaming, control flow — with the *quantum-writing* parts expressed as reusable, transpilable, debuggable Qiskit `QuantumCircuit` macros.

## Who is this for?

- Experimentalists on Quantum Machines hardware who want Qiskit ergonomics with QUA power underneath.
- Researchers building **hybrid classical–quantum control loops** and streaming-heavy workloads.
- Developers who need a documented bridge between Qiskit abstractions and QOP/QUA.

Start with **Providers** and **Backend & Utilities**, then **Primitives** and **Parameter Table** as you move toward hybrid programs.

## Experimental project — feedback welcome

This provider is **experimental by design**. If you have suggestions, use cases, or integration issues, contact **Arthur Strauss** at **arthur.strauss@u.nus.edu**.
