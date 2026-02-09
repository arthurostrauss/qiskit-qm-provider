# Examples

This folder contains standalone example workflows for the qiskit-qm-provider.

| Example | Description |
|--------|-------------|
| [sampler_workflow.py](sampler_workflow.py) | Run circuits with `QMSamplerV2` and the IQCC provider. |
| [estimator_workflow.py](estimator_workflow.py) | Run expectation-value jobs with `QMEstimatorV2` and real-time parameter input. |
| [custom_gate.py](custom_gate.py) | Add a custom parametric gate to the backend Target and sync with the QUA compiler. |
| [circuit_calibrations_pulse.py](circuit_calibrations_pulse.py) | Attach Qiskit Pulse calibrations to a circuit via `add_calibration` and run with the backend. |
| [iqcc_t1_experiment.py](iqcc_t1_experiment.py) | Run a Qiskit Experiments T1 characterization using a backend from `IQCCProvider`. |

**Note:** Examples that use `IQCCProvider` require a valid API token and access to the IQCC platform. Replace placeholder backend names (e.g. `"qolab"`, `"arbel"`) and paths with your own as needed.
