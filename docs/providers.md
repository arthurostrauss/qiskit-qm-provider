# Providers

## QMProvider

`qiskit_qm_provider.providers.qm_provider.QMProvider`

The standard provider for connecting to a local or server-based Quantum Orchestration Platform where the Quam state is stored locally.

### `__init__(state_folder_path: Optional[str] = None)`
Initializes the provider.
- `state_folder_path`: Path to the local Quam state folder.

### `get_backend(machine: Optional[QuamRoot] = None, **backend_options) -> QMBackend`
Returns a `QMBackend` instance.
- `machine`: Optional Quam instance. If None, it loads from `state_folder_path`.
- `**backend_options`: Options passed to the backend.

### `get_machine() -> Quam`
Loads and returns the latest Quam state from the `state_folder_path`.

---

## QmSaasProvider

`qiskit_qm_provider.providers.qm_saas_provider.QmSaasProvider`

Provider for connecting to the Quantum Machines SaaS platform.

### `__init__(email: Optional[str], password: Optional[str], host: Optional[str], version: Optional[str])`
Initializes the SaaS provider.
- `email`, `password`, `host`: Credentials for the SaaS platform. If None, attempts to read from `~/qm_saas_config.json`.
- `version`: Optional QOP version.

### `get_backend(quam_state_folder_path: Optional[str] = None, simulation_config: Optional[SimulationConfig] = None) -> QMBackend`
Returns a `QMBackend` instance connected to a SaaS instance.
- `quam_state_folder_path`: Path to the Quam state.
- `simulation_config`: Configuration for simulation.

---

## IQCCProvider

`qiskit_qm_provider.providers.iqcc_cloud_provider.IQCCProvider`

Provider for accessing devices at the Israeli Quantum Computing Center (IQCC).

### `__init__(api_token: Optional[str] = None)`
Initializes the IQCC provider.
- `api_token`: API token for IQCC authentication.

### `get_backend(name: str | IQCCQuam) -> FluxTunableTransmonBackend`
Returns a backend for the specified machine name.
- `name`: Name of the quantum computer (e.g., "arbel") or a Quam instance.
