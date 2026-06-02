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
sphinx-build -b html docs docs/_build/html
```

The first pass generates autosummary stub pages under `docs/apidocs/stubs/` (gitignored). Run the command a second time — or add `-W` on the second pass — to treat warnings as errors once stubs exist:

```bash
sphinx-build -b html docs docs/_build/html
sphinx-build -b html -W docs docs/_build/html
```

Open `docs/_build/html/index.html` in your browser.

## Contents

- [Workflows](workflows.md) — routing guide for main use paths.
- [Providers](providers.md) — local, SaaS, and IQCC providers.
- [Backend](backend.md) — `QMBackend`, embedding, utilities.
- [Primitives](primitives.md) — QOP-aware Sampler and Estimator.
- [Parameter Table](parameter_table.md) — hybrid classical–quantum data flow.
- [Error-Correction Workflow](error_correction.md) — EC pattern with `ParameterTable`.
- [API Reference](apidocs/qm.rst) — autodoc module index.
