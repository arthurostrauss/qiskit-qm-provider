# API Documentation

This directory contains documentation for `qiskit-qm-provider`, built with **Sphinx** and the **Qiskit ecosystem theme** (`qiskit-ecosystem`).

## Two layers

- **Guides** (`.md` pages) — purpose, architecture, and practical snippets.
- **API Reference** (`apidocs/`) — auto-generated signatures and docstrings via `autodoc` / `autosummary`.

## Local environment

Use the project virtual environment:

```bash
source ~/venvs/rl_qoc/bin/activate
pip install -e ".[docs]"
```

## Build locally

From the repository root:

```bash
sphinx-build -b html -W docs docs/_build/html
```

Autosummary generates stub pages under `docs/apidocs/stubs/` during the build (gitignored). Class and function stubs use `:members:` so docstrings, methods, and attributes render on each API page.

Open `docs/_build/html/index.html` in your browser.

## Contents

- [Workflows](workflows.md) — routing guide for main use paths.
- [Providers](providers.md) — local, SaaS, and IQCC providers.
- [Backend](backend.md) — `QMBackend`, embedding, utilities.
- [Measurement outputs](measurement_outputs.md) — `comp.outputs`, compilation-local handles.
- [Primitives](primitives.md) — QOP-aware Sampler and Estimator.
- [Parameter Table](parameter_table.md) — hybrid classical–quantum data flow.
- [Error-Correction Workflow](error_correction.md) — EC pattern with `ParameterTable`.
- [API Reference](apidocs/qm.rst) — autodoc module index.
