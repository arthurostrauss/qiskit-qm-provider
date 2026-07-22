# Providers

Providers are **environment adapters**: they answer "how do I obtain a [`QMBackend`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) for my setup?" while remaining hardware-agnostic through bring-your-own `QuamRoot` and [`QMBackend`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) subclasses.

For method signatures and constructor parameters, see the [Providers API reference](apidocs/qm_providers.rst).

## Purpose

Each provider connects a specific execution environment (local QOP, cloud simulator, or IQCC) to the same backend interface. Once you have a backend, workflows are identical: transpile circuits, then use `backend.run()`, primitives, or hybrid embedding.

The default QuAM class when none is supplied is **`FluxTunableQuam`** from [quam-builder](https://github.com/Quantum-Machines/quam-builder). The default backend class is the base [`QMBackend`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst). Custom labs should pass explicit `quam_cls` and `backend_cls` matching their hardware.

## Seeding gate macros with add_basic_macros

After obtaining a backend, populate standard operations with [`add_basic_macros`](apidocs/stubs/qiskit_qm_provider.quam_macros.superconducting.add_basic_macros.rst):

```python
from qiskit_qm_provider import QMProvider, add_basic_macros

provider = QMProvider(state_folder_path="/path/to/quam/state")
backend = provider.get_backend()
add_basic_macros(backend)
```

**Important:** these macros (`x`, `sx`, `rz`, `sy`, `sydg`, `measure`, `reset`, `delay`, `id`, `cz`) are **currently tailored to flux-tunable transmon** topology. They assume pulse and macro naming conventions from `FluxTunableQuam` / quam-builder (e.g. `x180`, `x90`, readout pulses, `CZGate` on pairs). They are a **starting point**, not a universal hardware definition. Users on other platforms should override or replace macros on their own `QuamRoot`, and coordinate with the **Quantum Machines team** for appropriate [quam-builder](https://github.com/Quantum-Machines/quam-builder) extensions for their architecture.

The function lives under `qiskit_qm_provider.quam_macros.superconducting` and is also re-exported from the top-level package and `qiskit_qm_provider.backend.backend_utils`. Extra keyword arguments are forwarded to `ResetMacro` via `**reset_macro_kwargs`.

## [`QMProvider`](apidocs/stubs/qiskit_qm_provider.providers.QMProvider.rst) — local QOP

Use [`QMProvider`](apidocs/stubs/qiskit_qm_provider.providers.QMProvider.rst) when you have a local QOP stack and a QuAM state folder on disk.

```python
from qiskit_qm_provider import QMProvider, FluxTunableTransmonBackend

# Custom QuAM + backend
provider = QMProvider(state_folder_path="/path/to/quam/state", quam_cls=MyCustomQuam)
backend = provider.get_backend(backend_cls=MyBackend)

# Flux-tunable transmon (explicit)
provider = QMProvider(state_folder_path="/path/to/quam/state")
backend = provider.get_backend(backend_cls=FluxTunableTransmonBackend)
```

Access the underlying machine via `backend.machine`.

## [`QmSaasProvider`](apidocs/stubs/qiskit_qm_provider.providers.qm_saas_provider.QmSaasProvider.rst) — cloud simulation

Use [`QmSaasProvider`](apidocs/stubs/qiskit_qm_provider.providers.qm_saas_provider.QmSaasProvider.rst) for QM's cloud simulator. Requires `pip install qiskit-qm-provider[qm_saas]`.

```python
from qiskit_qm_provider import QmSaasProvider, FluxTunableTransmonBackend

provider = QmSaasProvider(email="...", password="...", host="...")
backend = provider.get_backend(
    quam_state_folder_path="/path/to/quam/state",
    backend_cls=FluxTunableTransmonBackend,
)
```

Credentials can be read from `~/qm_saas_config.json` when omitted.

## [`IQCCProvider`](apidocs/stubs/qiskit_qm_provider.providers.iqcc_cloud_provider.IQCCProvider.rst) — IQCC cloud devices

Use [`IQCCProvider`](apidocs/stubs/qiskit_qm_provider.providers.iqcc_cloud_provider.IQCCProvider.rst) for devices at the Israeli Quantum Computing Center. Requires `pip install qiskit-qm-provider[iqcc]`. Always returns [`FluxTunableTransmonBackend`](apidocs/stubs/qiskit_qm_provider.backend.FluxTunableTransmonBackend.rst).

This is a common entry point for **Qiskit Experiments** characterization — see the [Qiskit Experiments caveats on the home page](index.md#using-qiskit-experiments-with-this-provider).

```python
from qiskit_qm_provider import IQCCProvider

provider = IQCCProvider(api_token="...")
backend = provider.get_backend(
    "arbel",
    quam_state_folder_path="/path/to/quam/state",  # or QUAM_STATE_PATH
)
```

## Comparison

| Provider | Environment | Extra install | Backend type | Typical use |
|----------|-------------|---------------|--------------|-------------|
| [`QMProvider`](apidocs/stubs/qiskit_qm_provider.providers.QMProvider.rst) | Local QOP + on-disk QuAM | — | User-chosen (default [`QMBackend`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst)) | Lab hardware |
| [`QmSaasProvider`](apidocs/stubs/qiskit_qm_provider.providers.qm_saas_provider.QmSaasProvider.rst) | QM cloud simulator | `[qm_saas]` | User-chosen | Cloud simulation |
| [`IQCCProvider`](apidocs/stubs/qiskit_qm_provider.providers.iqcc_cloud_provider.IQCCProvider.rst) | IQCC remote devices | `[iqcc]` | [`FluxTunableTransmonBackend`](apidocs/stubs/qiskit_qm_provider.backend.FluxTunableTransmonBackend.rst) | IQCC + Experiments |

## Related

- **Guide:** [Workflows — standard execution](workflows.md#running-qiskit-circuits-on-qm-hardware-or-simulators)
- **API:** [Providers reference](apidocs/qm_providers.rst)
- **Examples:** `examples/sampler_workflow.py`, `examples/iqcc_t1_experiment.py`
