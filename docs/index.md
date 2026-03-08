---
title: Home
layout: default
nav_order: 1
has_children: true
---

# Qiskit QM Provider

Welcome — this project is a **toolbox for crossing abstraction layers** between high-level Qiskit circuits and low-level QUA control programs on Quantum Machines' Quantum Orchestration Platform (QOP).

Thanks to Qiskit's fully integrated stack, and our close integration with QOP's latest real‑time capabilities, this provider is meant to **push state‑of‑the‑art use cases**: hybrid programs that combine classical and quantum workloads, while staying in familiar Qiskit terms.

## What this toolbox gives you

- **A bridge between abstractions**: from Qiskit circuits, primitives, and (where applicable) Pulse, down to QUA programs and QuAM‑defined hardware.
- **Minimal friction between layers**: reuse the same circuits and observables across:
  - standard Qiskit workflows (`backend.run()`, `QMSampler`, `QMEstimator`)
  - real‑time QUA programs with streaming parameters and control flow.
- **Customizable hardware model**: bring your own `QuamRoot` subclass and `QMBackend` subclass, or start from the provided `FluxTunableTransmonBackend`.

In short: you design algorithms and calibrations with Qiskit, and this toolbox takes care of the messy parts of talking to QOP and QUA.

## Installation

```bash
pip install qiskit-qm-provider
```

For IQCC cloud access and QM SaaS simulation:

```bash
pip install qiskit-qm-provider[iqcc]
pip install qiskit-qm-provider[qm_saas]
```

## Where to go next

Use the left sidebar (under **Home**) to browse the main components:

- **[Providers](providers.md)** – local QMProvider, cloud IQCCProvider, and QmSaasProvider:
  how to obtain a backend for your environment (lab server, SaaS simulator, or IQCC).
- **[Backend & Utilities](backend.md)** – `QMBackend`, `FluxTunableTransmonBackend`,
  `QMInstructionProperties`, and helpers like `add_basic_macros` and `get_measurement_outcomes`.
- **[Primitives](primitives.md)** – `QMEstimatorV2` and `QMSamplerV2`, wired to QOP's real‑time
  streaming and control‑flow abilities.
- **[Parameter Table](parameter_table.md)** – `ParameterTable` and `Parameter`, the core layer
  for expressing real‑time parameters and classical data in hybrid QUA/Qiskit workloads.

Example workflows (calibrations, primitives, IQCC + Qiskit Experiments) live in the
[examples](https://github.com/arthurostrauss/qiskit-qm-provider/tree/main/examples) folder.

## Design philosophy

Working at the intersection of **algorithms** and **control** is hard: each abstraction layer
solves a different problem (circuit design, transpilation, scheduling, hardware control), but in
practice you often need to touch several layers at once. This provider is designed to:

- keep the **Qiskit user experience** for algorithm design and experimentation,
- expose **QOP/QUA features** (real‑time feedback, streaming, hybrid control) without requiring
  you to rewrite everything as low‑level QUA code,
- make it easy to **embed Qiskit circuits inside larger QUA programs**, using the same
  calibration model and parameter interface.

If you are building experiments that mix calibration, characterization, optimal control, and
algorithmic workloads, this toolbox is meant to minimize the glue you need to write yourself.

## Typical workflows

Here are some common ways to use the provider; each one is documented in more detail in the
sidebar pages:

- **Run Qiskit circuits on QM hardware or simulators**
  - Use `QMProvider` or `QmSaasProvider` to get a backend.
  - Call `backend.run()` or use `QMSamplerV2` / `QMEstimatorV2`.

- **Calibrate and characterize hardware with Qiskit Experiments and IQCC**
  - Use `IQCCProvider` (with the `[iqcc]` extra) to access IQCC devices.
  - Combine with Qiskit Experiments for T1/T2/RB/etc. while QOP handles the low‑level control.

- **Embed Qiskit circuits inside QUA programs**
  - Use `backend.quantum_circuit_to_qua(...)` to turn a `QuantumCircuit` into a QUA macro.
  - Use `ParameterTable` and `get_measurement_outcomes` to shuttle data and parameters between
    QUA and Python.

- **Define custom gates and hardware‑specific backends**
  - Extend `QMBackend` or start from `FluxTunableTransmonBackend`.
  - Use `QMInstructionProperties` to attach QUA macros to new instructions in the Qiskit Target.

Each of these paths is meant to feel like a natural extension of Qiskit, while still exposing
the full power of QOP and QUA underneath.

## Who is this for?

- Experimentalists running Quantum Machines hardware who want to:
  - stay in Qiskit for most of their workflow,
  - experiment with new calibration / control strategies in QUA,
  - or combine both without duplicating logic.
- Researchers exploring **hybrid classical–quantum control loops**, real‑time feedback, and
  streaming‑heavy workloads.
- Developers who want a **clean, documented interface** between Qiskit abstractions and
  QOP/QUA, without re‑implementing that glue in every project.

If that describes you, start with **Providers** and **Backend & Utilities** in the sidebar,
then move on to **Primitives** and **Parameter Table** as you begin to build more advanced
hybrid programs.
