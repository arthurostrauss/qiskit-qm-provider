# Providers

## QMProvider

`qiskit_qm_provider.providers.qm_provider.QMProvider`

The standard provider for connecting to a local or server-based Quantum Orchestration Platform where the QuAM state is stored locally.

`QMProvider` is **hardware-agnostic**: users supply their own `QuamRoot` subclass (via `quam_cls`) and their own `QMBackend` subclass (via `backend_cls`) to match their specific hardware.  When these are omitted, the provider falls back to `FluxTunableQuam` (from *quam-builder*) and the base `QMBackend`, respectively.

### `__init__(state_folder_path: Optional[str] = None, quam_cls: Type[QuamRoot] | None = None)`
Initializes the provider.
- `state_folder_path`: Path to the local QuAM state folder.
- `quam_cls`: `QuamRoot` subclass to use for loading the machine.  Defaults to `FluxTunableQuam` from *quam-builder*.

### `get_backend(machine: Optional[QuamRoot] = None, backend_cls: Type[QMBackend] | None = None, **backend_options) -> QMBackend`
Returns a `QMBackend` (or subclass) instance.
- `machine`: Optional pre-loaded QuAM instance.  If `None`, loads from `state_folder_path`.
- `backend_cls`: `QMBackend` subclass to instantiate (e.g. `FluxTunableTransmonBackend` or a custom subclass).  Defaults to the base `QMBackend`.
- `**backend_options`: Options passed to the backend constructor (e.g. `qmm`, `name`, `shots`, `simulate`).

### `get_machine() -> QuamRoot`
Loads and returns the latest QuAM state from the `state_folder_path`.

---

## QmSaasProvider

`qiskit_qm_provider.providers.qm_saas_provider.QmSaasProvider`

Provider for connecting to the Quantum Machines SaaS simulation platform.

Requires the `qm-saas` extras: `pip install qiskit-qm-provider[qm-saas]`.

### `__init__(email: Optional[str], password: Optional[str], host: Optional[str], version: Optional[str])`
Initializes the SaaS provider.
- `email`, `password`, `host`: Credentials for the SaaS platform.  If `None`, attempts to read from `~/qm_saas_config.json`.
- `version`: Optional QOP version string.

### `get_backend(quam_state_folder_path=None, simulation_config=None, quam_cls=None, backend_cls=None) -> QMBackend`
Returns a `QMBackend` (or subclass) instance connected to a SaaS simulator.
- `quam_state_folder_path`: Path to the QuAM state.
- `simulation_config`: Simulation configuration (defaults to 10 000 clock cycles).
- `quam_cls`: `QuamRoot` subclass for the machine.
- `backend_cls`: `QMBackend` subclass to instantiate.  Defaults to the base `QMBackend`.

---

## IQCCProvider

`qiskit_qm_provider.providers.iqcc_cloud_provider.IQCCProvider`

Provider for accessing devices at the Israeli Quantum Computing Center (IQCC).

Requires the `iqcc` extras: `pip install qiskit-qm-provider[iqcc]`.

IQCC backends are flux-tunable transmon machines; the provider always returns a `FluxTunableTransmonBackend`.

### `__init__(api_token: Optional[str] = None)`
Initializes the IQCC provider.
- `api_token`: API token for IQCC authentication.

### `get_machine(name: str, quam_state_folder_path: Optional[str] = None, quam_cls: Type[QuamRoot] | None = None) -> QuamRoot`
Fetches the latest QuAM state for the given IQCC device and returns a loaded QuAM instance.
- `name`: Name of the quantum computer (e.g., `"arbel"`).
- `quam_state_folder_path`: Optional path to the QuAM state folder. If omitted, the provider falls back to the `QUAM_STATE_PATH` environment variable. Supplying an explicit path is recommended for tests and non-interactive environments.
- `quam_cls`: Optional `QuamRoot` subclass to use for loading the machine. If omitted, the provider defaults to the IQCC-specific QuAM implementation from `iqcc_calibration_tools` (when available), and otherwise falls back to the standard `FluxTunableQuam` from *quam-builder*.

### `get_backend(name: str | QuamRoot, quam_state_folder_path: Optional[str] = None, quam_cls: Type[QuamRoot] | None = None) -> FluxTunableTransmonBackend`
Returns a `FluxTunableTransmonBackend` for the specified IQCC device or a pre-loaded QuAM instance.
- `name`: Either the name of the quantum computer (e.g., `"arbel"`) or a pre-loaded QuAM instance.
- `quam_state_folder_path`: Optional path to the QuAM state folder when `name` is a string. If omitted, falls back to the `QUAM_STATE_PATH` environment variable.
- `quam_cls`: Optional `QuamRoot` subclass to use when loading the machine. If omitted, the same default resolution as in `get_machine` is used (IQCC-specific QuAM from `iqcc_calibration_tools` when present, otherwise `FluxTunableQuam` from *quam-builder*).
