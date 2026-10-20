# Examples

This folder contains standalone example workflows for the qiskit-qm-provider.

| Example | Description |
|--------|-------------|
| [sampler_workflow.py](sampler_workflow.py) | Run circuits with `QMSamplerV2` and the IQCC provider. |
| [estimator_workflow.py](estimator_workflow.py) | Run expectation-value jobs with `QMEstimatorV2` and real-time parameter input. |
| [custom_gate.py](custom_gate.py) | Add a custom parametric gate to the backend Target and sync with the QUA compiler. |
| [circuit_calibrations_pulse.py](circuit_calibrations_pulse.py) | Attach Qiskit Pulse calibrations to a circuit via `add_calibration` and run with the backend. |
| [iqcc_t1_experiment.py](iqcc_t1_experiment.py) | Run a Qiskit Experiments T1 characterization using a backend from `IQCCProvider`. |
| [qm_saas_sampler.py](qm_saas_sampler.py) | Simulate circuits on the QM cloud simulator with `QmSaasProvider` and `QMSamplerV2`. |

**Note:** Examples that use `IQCCProvider` require a valid API token and access to the IQCC platform. Replace placeholder backend names (e.g. `"qolab"`, `"arbel"`) and paths with your own as needed.

Examples that use `QmSaasProvider` require QM SaaS credentials and `qm-saas>=1.2.0` (installed by `pip install qiskit-qm-provider[qm-saas]`), which provides the TLS-secured connection required by the cloud simulator from QOP 3.8.1 onwards.
