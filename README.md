# Qiskit QM Provider

**A comprehensive interface for tight integration between the Qiskit ecosystem and Quantum Machine's Quantum Orchestration Platform (QOP).**

## Overview

The `qiskit-qm-provider` repository proposes a tight integration between the Qiskit ecosystem and QUA, the proprietary language of quantum machines for the Quantum Orchestration Platform. It is designed to leverage the latest real-time processing features of QOP while maintaining the ease of use of Qiskit for high-level quantum algorithm design.

The goal of this provider is to explain the intended usage of components that bridge the gap between abstract quantum circuits and hardware execution, featuring:

1.  **Quam Integration**: A Qiskit backend implementation of the [Quam structure](https://qua-platform.github.io/quam/), enabling automated fetching of basis gates, coupling maps, and other key properties. This facilitates the use of the entire Qiskit transpilation pipeline by breaking down high-level algorithms into circuits readily executable on hardware.
2.  **Specialized Providers**: Support for three different execution environments (Local, SaaS, IQCC).
3.  **Real-time Primitives**: Custom implementations of Qiskit Primitives (`Estimator` and `Sampler`) optimized for QOP capabilities like real-time parameter updates and control flow.

## Providers

We support different integrations available through three different providers. Users can obtain a backend directly from the provider. The underlying Quam instance is accessible via the `backend.machine` attribute.

It is also possible to populate the machine with standard operations (like `x`, `sx`, `rz`, `measure`, `reset`, `cz`) using the `add_basic_macros` utility.

```python
from qiskit_qm_provider.backend.backend_utils import add_basic_macros
# After getting backend:
# add_basic_macros(backend)
```

1.  **QMProvider**: Assumes the experimentalist has a Quantum Orchestration Platform directly accessible on their server and a local Quam instance stored on the computer.
    ```python
    from qiskit_qm_provider import QMProvider
    provider = QMProvider(state_folder_path="/path/to/quam/state")
    backend = provider.get_backend()
    ```

2.  **QmSaasProvider**: Connects directly to the [QM SaaS platform](https://docs.quantum-machines.co/latest/docs/Guides/qm_saas_guide/).
    ```python
    from qiskit_qm_provider import QmSaasProvider
    provider = QmSaasProvider(email="...", password="...", host="...")
    backend = provider.get_backend(quam_state_folder_path="...")
    ```

3.  **IQCCProvider**: Provides access to available devices at the Israeli Quantum Computing Center (IQCC) in Tel Aviv.
    ```python
    from qiskit_qm_provider import IQCCProvider
    provider = IQCCProvider(api_token="...")
    backend = provider.get_backend("arbel") # Example machine name
    ```

## Qiskit Primitives on QOP

We provide custom implementations of the standard Qiskit Primitives, `QMEstimator` and `QMSampler`, which are straightforward adaptations of the [standard Qiskit primitives](https://quantum.cloud.ibm.com/docs/en/guides/primitives). They leverage the core capabilities of the Quantum Orchestration Platform to optimize execution through:

1.  **Real-time Parameter Adjustment**: The ability to adjust parameter values in real-time and load them asynchronously using **Input Streaming** or **DGX Quantum**.
2.  **Real-time Control Flow**: The ability to perform real-time control flow to estimate different expectation values seamlessly across a single compilation of a quantum circuit (specifically for the Estimator primitive).

### Usage Example

```python
from qiskit_qm_provider import QMEstimatorV2, QMEstimatorOptions, InputType

# Initialize Estimator with Input Streaming for real-time parameter updates
options = QMEstimatorOptions(input_type=InputType.INPUT_STREAM)
estimator = QMEstimatorV2(backend=backend, options=options)

# Run estimator job
job = estimator.run([(circuit, observables, parameter_values)])
result = job.result()
```

We also implement the traditional `backend.run()` function, which closely mimics the `Sampler` primitive behavior.

## Hybrid QUA and Qiskit Interface

We envision this tool as more than just a Qiskit bridge; it is a new interface to intertwine the power of Qiskit and QUA. You can build QUA programs over many qubits that incorporate the execution of Qiskit quantum circuits as QUA-embedded macros.

### Workflow: Embedding and Processing

When embedding Qiskit circuits into QUA programs, the typical workflow involves two steps using `backend.quantum_circuit_to_qua()` and `get_measurement_outcomes()`:

1.  **`backend.quantum_circuit_to_qua(qc, ...)`**: This function compiles the Qiskit circuit into QUA instructions and inserts them into the current QUA program context. It returns a result object.
2.  **`get_measurement_outcomes(qc, result)`**: This utility function takes the circuit and the result from the previous step. It returns a dictionary containing QUA variables corresponding to the measurement outcomes.
    *   It provides the raw measurement value.
    *   Crucially, it computes the **`state_int`**, a QUA integer variable representing the measurement result of a classical register.

This `state_int` is essential for **real-time control flow** and **data streaming**, allowing you to use Qiskit for complex circuit definitions while leveraging QUA for fast feedback logic.

### Example: Embedding Qiskit Circuits in QUA

```python
from qm.qua import program
from qiskit_qm_provider import ParameterTable

# ... Define Qiskit circuit 'qc' ...

with program() as prog:
    # Embed the Qiskit circuit as a QUA macro
    # param_table allows passing real-time QUA variables to the circuit parameters
    backend.quantum_circuit_to_qua(qc, param_table=my_param_table)
```

### Error Correction and Parameter Table

For scalable error correction workflows, where hybrid classical-quantum computing is essential, we introduce the **Parameter Table**. This module provides a full interface to express parametric programs and seamless communication between a client (or DGX server) and the QUA program.

Below is an example of an error correction workflow where data handling is critical. This showcases how to deal with parameter wake workflows when Qiskit cannot save data on the fly but must store it in new memory slots for each syndrome declaration. Note the use of `get_measurement_outcomes` to extract the syndrome state for feedback.

```python
from qm.qua import *
from qiskit_qm_provider import Parameter, ParameterTable, ParameterPool, Direction, InputType, QUA2DArray
from qiskit_qm_provider.backend.backend_utils import get_measurement_outcomes, add_basic_macros
from qiskit import transpile

# ... (Assume backend, syndrome_circuit, recovery_circuit, encoding_circuit are defined) ...
add_basic_macros(backend)

ParameterPool.reset()
num_cycles = 2
memory_exp_length = 50
d = 3
input_type = InputType.INPUT_STREAM

# Define parameters and tables
syndrome_data: Parameter = Parameter("syndrome_data", 0, input_type=input_type, direction=Direction.INCOMING)
recovery_vars: ParameterTable = ParameterTable.from_qiskit(recovery_circuit, input_type=input_type)

syndrome_circuit = transpile(syndrome_circuit, backend)
recovery_circuit = transpile(recovery_circuit, backend)
encoding_circuit = transpile(encoding_circuit, backend)

ancilla_creg = syndrome_circuit.cregs[0]

with program() as qec_prog:
    state_int = declare(int, value=0)
    m = declare(int)
    round = declare(int)

    # Declare variables for parameters
    recovery_vars.declare_variables()
    syndrome_data.declare_variable()
    syndrome_data.declare_stream()

    if backend.init_macro:
        backend.init_macro()

    with for_(m, 0, m < memory_exp_length, m + 1):
        with for_(round, 0, round < num_cycles, round + 1):
            # Execute syndrome measurement circuit converted to QUA
            syndrome_meas_result = backend.quantum_circuit_to_qua(syndrome_circuit)

            # Extract measurement outcomes for real-time processing
            syndrome_meas_result_meas = get_measurement_outcomes(syndrome_circuit, syndrome_meas_result)
            state_int_val = syndrome_meas_result_meas[ancilla_creg.name]["state_int"]

            # Update syndrome data parameter and stream back
            syndrome_data.assign(state_int_val)
            syndrome_data.stream_back(reset=True)

        # Load recovery variables (simulating feedback latency/calculation)
        recovery_vars.load_input_values()
        # Execute recovery circuit with updated parameters
        recovery_circuit_result = backend.quantum_circuit_to_qua(recovery_circuit, recovery_vars)

    if input_type != InputType.DGX_Q:
        with stream_processing():
            syndrome_data.stream_processing()
```

## Parameter Table API Documentation

The `ParameterTable` is a core component for managing real-time parameters.

### `ParameterTable`

Class enabling the mapping of parameters to be updated to their corresponding "to-be-declared" QUA variables.

#### Initialization
```python
ParameterTable(parameters_dict, name=None)
```
- `parameters_dict`: Dictionary `{ "name": (initial_value, qua_type, input_type, direction) }` or list of `Parameter` objects.
- `name`: Optional name for the table.

#### Methods

- **`declare_variables(pause_program=False)`**: QUA Macro to declare all QUA variables associated with the table.
- **`load_input_values(filter_function=None)`**: QUA Macro to load input values from the input stream/IO/DGX.
- **`push_to_opx(param_dict, job, qm, verbosity)`**: Client function to push values to the OPX.
- **`fetch_from_opx(job, fetching_index, fetching_size)`**: Client function to fetch values from the OPX.
- **`stream_back(reset=False)`**: QUA Macro to stream values back to the client/server.
- **`from_qiskit(qc, input_type, filter_function)`**: Class method to create a table from a Qiskit QuantumCircuit's parameters.

### `Parameter`

Represents a single parameter mapped to a QUA variable.

- **`assign(value)`**: QUA Macro to assign a value to the parameter's QUA variable.
- **`save_to_stream()`**: QUA Macro to save the current value to its output stream.

## Compatibility and Custom Calibrations

This provider is compatible with both **Qiskit 1.x** and **Qiskit 2.x**.

### Qiskit 1.x (Pulse Support)
Sticking to Qiskit 1.x enables partial support for **Qiskit Pulse**, allowing custom pulse-level calibrations expressed in Qiskit Pulse to be directly translated into a QUA macro.

### Qiskit 2.x (Qiskit Pulse Deprecation)
We encourage the adoption of Qiskit 2.0. The novel way to express custom calibrations is through `QMInstructionProperties`. This allows you to specify additional gates in the backend target that contain a customized QUA macro.

#### Example: Parametric CNOT Gate with Custom QUA Macro

This example demonstrates how to add a custom parametric gate to the hardware backend using `QMInstructionProperties`.

```python
from qiskit.circuit import QuantumCircuit, Parameter as QiskitParameter, Gate
from qiskit.transpiler import Target
from qiskit_qm_provider import QMProvider, QMInstructionProperties
from qiskit_qm_provider.backend.backend_utils import add_basic_macros
from qm.qua import *

# 1. Define Parametric CNOT gate (Simulation)
# Noise is embedded in the gate definition as a rotation parameter
num_params = 1
θ = [QiskitParameter(f"θ_{i}" if num_params > 1 else "θ") for i in range(num_params)]
cx_circ = QuantumCircuit(2, name="cx_cal")
cx_circ.cx(0, 1)  # Ideal gate
cx_circ.rx(θ[0], 1)  # Coherent noise structure

# Define the custom gate for each CNOT
θ_list = [[QiskitParameter(f"θ_{i}_{j}" if num_params > 1 else f"θ_{i}") for j in range(num_params)] for i in range(4)]
cx_gates = {f"cx_{i}": Gate(f"cx_{i}", 2, θ_list[i]) for i in range(4)}
for i in range(4):
    cx_gates[f"cx_{i}"].definition = cx_circ.assign_parameters(θ_list[i], inplace=False)

# 2. Setup Provider and Backend
provider = QMProvider("/path/to/quam/state")
backend = provider.get_backend()
# Access the machine instance if needed, e.g., to add basic macros
add_basic_macros(backend)


# 3. Add Custom Gates to Hardware Backend
# We inform the transpiler that those custom gates shall be complemented by a specific
# custom pulse level calibration (QUA macro).

qubit_pairs_indices = list(backend.coupling_map.get_edges())
instruction_props = {cx: {} for cx in cx_gates}

# In this example, we define a simple parametric macro for demonstration.
# In a real scenario, this would be a calibrated pulse sequence.
for i, qubit_pair_index in enumerate(qubit_pairs_indices):
    qubit_pair = backend.get_qubit_pair(qubit_pair_index)
    for cx_name in cx_gates:
        # Define the QUA macro (parametric amplitude for CZ)
        macro = lambda amp: qubit_pair.apply("cz", amplitude_scale=amp)

        # Create QMInstructionProperties with the QUA macro
        qm_prop = QMInstructionProperties(qua_pulse_macro=macro)

        # Update the backend target
        # Note: In a full script you would attach this to the target properly
        # backend.target.update_instruction_properties(cx_name, qubit_pair_index, qm_prop)
```

## Installation

```bash
pip install qiskit-qm-provider
```

## License

This project is licensed under the MIT License.
