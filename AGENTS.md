# AGENTS.md

## Cursor Cloud specific instructions

### What this is
`qiskit-qm-provider` is a **pure Python library** (no server/web app): a Qiskit provider that
compiles Qiskit circuits/pulses into QUA for Quantum Machines' Quantum Orchestration Platform.
The core entry point is `QMBackend.quantum_circuit_to_qua()` (Qiskit circuit → OpenQASM 3 → QUA).

### Environment
- Managed by `uv`; the virtualenv lives at `.venv` (Python 3.12). Run tools via
  `.venv/bin/python ...` or `uv run ...`. `uv` is installed under `$HOME/.local/bin`.
- The startup update script already installs everything (see below), so you normally do not
  need to reinstall anything.

### Non-obvious dependency gotchas (important)
- **The committed `uv.lock` is internally inconsistent.** It pins `quam==0.4.1` together with
  `qm-qua==1.2.6`, but `quam` 0.4.1 imports `AmpValuesType` from `qm.qua._dsl`, which `qm-qua`
  1.2.6 removed — so a plain `uv sync` yields an env that **cannot import the package**. The fix
  (applied by the update script) is to install `quam==0.6.0` (allowed by the provider's
  `quam>=0.4.1` and quam-builder's `quam>=0.4.0`). Do not "fix" this by downgrading `qm-qua`.
- **`quam-builder` and `pytest` are not in `uv.lock`.** `quam-builder` is not on PyPI and is
  installed from git at `v0.4.0` (see `README.md` / `docs/installation.md`). `pytest` is a dev
  tool installed on top of the sync.
- Installing `quam-builder` upgrades `qualang-tools` to `0.23.0` (it requires `>=0.22.0`); this is
  expected and correct, even though the lock pins `0.19.5`.

### quarc / hardware caveats
- `quarc` (Open Acceleration Stack / QUARC) is a **private-alpha package, not on PyPI**, so
  `qiskit_qm_provider.QUARC_AVAILABLE` is `False` here. As a result, **2 tests fail** and cannot
  pass without QM's private package:
  `test/test_parameter_pool.py::TestParameterPoolFromQuarcModuleDict` (they call
  `ParameterPool.from_quarc_module*` without guarding for quarc's absence). Treat these two as
  expected failures in this environment; everything else passes.
- Hardware-backed test fixtures (`quam_machine`, `flux_tunable_backend`, `qm_provider` in
  `test/conftest.py`) require the `QUAM_STATE_PATH` env var to point at a QuAM state directory and
  are **skipped** when it is unset (~120 tests). You can also build a machine in memory with the
  quam-builder wirer (see the hello-world pattern below) instead of maintaining a state folder.

### How to test / build / run
- Tests: `.venv/bin/python -m pytest` (expected: ~294 passed, ~123 skipped, 2 quarc failures).
- Docs build (this repo's only "build" step, run with warnings-as-errors in CI):
  `.venv/bin/python -m sphinx -b html -W docs docs/_build/html`.
- **No linter is configured** in this repo (no ruff/black/flake8/mypy config in `pyproject.toml`).
- There is no application to "serve". To exercise the core pipeline end-to-end without hardware:
  build a `FluxTunableQuam` via `qualang_tools.wirer` + `quam_builder` (set `QUAM_STATE_PATH` to a
  temp dir so `machine.save()` works), get a `FluxTunableTransmonBackend` from `QMProvider`, seed
  gates with `add_basic_macros`, then `transpile(circuit, backend)` and
  `backend.quantum_circuit_to_qua(tqc)`. Note: quam-builder's default pulses are named
  `x180_DragGaussian` etc., but `add_basic_macros` expects canonical `x180`/`x90`/`y90`/`-y90`
  (and `readout`), so an in-memory demo machine must add those canonically-named operations.
