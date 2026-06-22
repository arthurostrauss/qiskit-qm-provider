# Installation

## Core package

Install `qiskit-qm-provider` from PyPI:

```bash
pip install qiskit-qm-provider
```

## quam-builder (required for built-in backends)

[quam-builder](https://github.com/qua-platform/quam-builder) provides the `FluxTunableQuam` and related QuAM components used by the built-in `FluxTunableTransmonBackend`. It is not published on PyPI and must be installed from source:

```bash
pip install git+https://github.com/qua-platform/quam-builder.git@v0.4.0
```

If you supply your own `QuamRoot` subclass and do not rely on `FluxTunableTransmonBackend`, this step is optional.

## Optional extras

### IQCC cloud access

Provides `IQCCProvider` for connecting to the Israeli Quantum Computing Center:

```bash
pip install qiskit-qm-provider[iqcc]
```

Requires `iqcc-cloud-client` and `jinja2` (both installed automatically via the extra).

### QM SaaS simulation

Provides `QmSaasProvider` for connecting to the [QM SaaS platform](https://docs.quantum-machines.co/latest/docs/Guides/qm_saas_guide/):

```bash
pip install qiskit-qm-provider[qm-saas]
```

### Documentation build

Dependencies for building these docs locally:

```bash
pip install qiskit-qm-provider[docs]
```

Then build with:

```bash
sphinx-build -b html -W docs docs/_build/html
```

## Open Acceleration Stack (OAS) and QUARC

Advanced real-time parameter workflows can be accelerated through Quantum Machines'
**Open Acceleration Stack (OAS)**, which features an **OPNIC** link enabling
high-bandwidth classical–quantum communication via **QUARC**.

QUARC is currently in a **private alpha** and is not publicly available on PyPI.
If you are interested in using these capabilities for advanced hybrid workflows,
please reach out to the [Quantum Machines team](https://www.quantum-machines.co/contact/).

## Python version support

`qiskit-qm-provider` requires **Python 3.10–3.12**. Python 3.13 is not yet supported.

## Verifying the installation

```python
import qiskit_qm_provider
print(qiskit_qm_provider.__version__)
```
