# API Documentation

This directory contains detailed API documentation for `qiskit-qm-provider`.
The docs are built with **Sphinx** and the **Qiskit ecosystem theme** (`qiskit-ecosystem`).

## Build locally

From the repository root, with your virtual environment activated:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in your browser to preview the site.

## Contents

- [Workflows and Examples](workflows.md): High-level workflows (local, SaaS, IQCC, hybrid embedding) with pointers to example scripts.
- [Providers](providers.md): Documentation for `QMProvider`, `QmSaasProvider`, and `IQCCProvider`.
- [Backend](backend.md): Documentation for `QMBackend` and utility functions like `add_basic_macros` and `get_measurement_outcomes`.
- [Primitives](primitives.md): Documentation for `QMEstimatorV2` and `QMSamplerV2`.
- [Parameter Table](parameter_table.md): Documentation for `ParameterTable` and `Parameter` classes.
- [Error‑Correction Workflow](error_correction.md): A didactic walkthrough of the hybrid error‑correction pattern and how `ParameterTable` helps.