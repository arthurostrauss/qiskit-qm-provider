# Qiskit QM Provider

**A comprehensive interface for tight integration between the Qiskit ecosystem and Quantum Machine's Quantum Orchestration Platform (QOP).**

## Installation

```bash
pip install qiskit-qm-provider
```

## Documentation

For full API documentation, please refer to the [docs folder](docs/). Example workflows (primitives, custom gates, calibrations, IQCC + Qiskit Experiments) are in the [examples](examples/) folder.

## Overview

The `qiskit-qm-provider` repository proposes a tight integration between the Qiskit ecosystem and QUA, the proprietary language of Quantum Machines for the Quantum Orchestration Platform. It is designed to leverage the latest real-time processing features of QOP while maintaining the ease of use of Qiskit for high-level quantum algorithm design.

The goal of this provider is to explain the intended usage of components that bridge the gap between abstract quantum circuits and hardware execution, featuring:

1. **Quam Integration**: A Qiskit backend implementation of the [Quam structure](https://qua-platform.github.io/quam/), enabling automated fetching of basis gates, coupling maps, and other key properties. This facilitates the use of the entire Qiskit transpilation pipeline by breaking down high-level algorithms into circuits readily executable on hardware.
2. **Specialized Providers**: Support for three different execution environments (Local, SaaS, IQCC).
3. **Real-time Primitives**: Custom implementations of Qiskit Primitives (`Estimator` and `Sampler`) optimized for QOP capabilities like real-time parameter updates and control flow.

## Providers

We support different integrations available through three different providers. Users can obtain a backend directly from the provider. The underlying Quam instance is accessible via the `backend.machine` attribute.

It is also possible to populate the machine with standard operations (like `x`, `sx`, `rz`, `measure`, `reset`, `cz`) using the `add_basic_macros` utility.

```python
from qiskit_qm_provider.backend.backend_utils import add_basic_macros
# After getting backend:
# add_basic_macros(backend)
```

1. **QMProvider**: Assumes the experimentalist has a Quantum Orchestration Platform directly accessible on their server and a local Quam instance stored on the computer.
  ```python
    from qiskit_qm_provider import QMProvider
    provider = QMProvider(state_folder_path="/path/to/quam/state")
    backend = provider.get_backend()
  ```
2. **QmSaasProvider**: Connects directly to the [QM SaaS platform](https://docs.quantum-machines.co/latest/docs/Guides/qm_saas_guide/).
  ```python
    from qiskit_qm_provider import QmSaasProvider
    provider = QmSaasProvider(email="...", password="...", host="...")
    backend = provider.get_backend(quam_state_folder_path="...")
  ```
3. **IQCCProvider**: Provides access to available devices at the Israeli Quantum Computing Center (IQCC) in Tel Aviv, Israel.
  ```python
    from qiskit_qm_provider import IQCCProvider
    provider = IQCCProvider(api_token="...")
    backend = provider.get_backend("arbel") # Example machine name
  ```

## Backends: QMBackend and FluxTunableTransmonBackend

The backends returned by the providers are the central interface that connects Qiskit to the Quantum Orchestration Platform. They serve two main roles: (1) representing the hardware in Qiskit’s terms, and (2) translating Qiskit circuits and schedules into QUA for execution.

### Representing the hardware in Qiskit

**QMBackend** (and its subclasses such as **FluxTunableTransmonBackend**) implement the interface needed to build the appropriate [Target](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.Target) object, which is the key abstraction used to represent a backend in Qiskit’s `BackendV2` model. The Target is populated from the existing [Quam](https://qua-platform.github.io/quam/) structure: the backend fetches **macros** (gate-level operations and their QUA implementations) from the machine’s qubits and qubit pairs, and derives the **coupling map** from the active qubit topology. This allows the full Qiskit transpilation pipeline to work (basis gates, connectivity, instruction properties) so that algorithms can be compiled down to circuits executable on the hardware.

### Circuit-to-QUA translation: qm_qasm and `quantum_circuit_to_qua`

Beyond the Target, the backend embeds Quantum Machines’ **qm_qasm** stack: a company-developed OpenQASM 3 → QUA compiler that turns Qiskit-exported circuits into QUA code. The main entry point is:

- `**backend.quantum_circuit_to_qua(qc, param_table=...)`**  
Compiles the Qiskit `QuantumCircuit` into QUA instructions and inserts them into the current QUA program context (when called inside a `with program():` block), or returns a compilation result that can be used to obtain a standalone QUA program. It is the direct path from Qiskit to QUA, without going through the primitives.

**ParameterTables and how parameters are supplied**

`quantum_circuit_to_qua` accepts a `param_table` argument that describes how symbolic and classical inputs are mapped to QUA. This is where the provider’s design diverges from standard Qiskit: parameters are not required to be bound at compile time; they can be bound **in real time** in QUA (e.g. as phase or amplitude of a pulse, or frame rotation). The tables are expected to fall into two conceptual categories:

1. **Symbolic (circuit) parameters → real-time QUA variables**
  Use `**ParameterTable.from_qiskit(qc, input_type=..., ...)`** to build a table from a circuit’s symbolic parameters. The table describes names and types for QUA variables that will hold values at runtime (e.g. loaded from an input stream, DGX Quantum, or set elsewhere in the QUA program). Those variables are assumed to be castable to real-time adjustable quantities (phase, amplitude, etc.). For more complex or custom workflows, consider reaching out to the maintainers.
  **Warning — parameter names:** The Quantum Orchestration Platform rejects parameter names that are not valid in its compilation pipeline. Qiskit often uses Greek letters or other non-ASCII symbols for symbolic parameters (e.g. `θ`, `φ`). When defining parameters that will be passed to `quantum_circuit_to_qua` or used with `ParameterTable.from_qiskit`, use **standard ASCII names** (e.g. `theta`, `phi`, `alpha`) so that the exported OpenQASM 3 and QUA compilation succeed.
2. **Classical input variables (Qiskit “input vars”)**
  Qiskit supports [real-time typed classical data](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.QuantumCircuit#working-with-real-time-typed-classical-data) via `Var` and input variables. These can represent values that are supplied from elsewhere in the QUA program or from a classical server. `ParameterTable.from_qiskit` can also incorporate these (symbolic and classical together), so a single table can feed both gate parameters and classical inputs into `quantum_circuit_to_qua`. That opens the door to **hybrid programs** (real-time feedback, adaptive circuits, classical control flow) in a way the traditional Qiskit circuit model does not natively support.

So: the “simple” path is the standard translation of Qiskit workflows through the primitives (`run()`, `QMSampler`, `QMEstimator`). The **extended** path is to embed circuits inside QUA via `quantum_circuit_to_qua` and ParameterTables, merging standard Qiskit with real-time QUA processing and hybrid workloads.

### FluxTunableTransmonBackend (and future hardware-specific backends)

**FluxTunableTransmonBackend** is a subclass of **QMBackend** for flux-tunable transmon machines. Hardware-specific backends like this add two things on top of the base backend:

1. **Hardware lifecycle and Quam integration**
  They pull from the [quam-builder](https://github.com/Quantum-Machines/quam-builder) interface (QM’s standardized product line for Quam-based configurations). For example, `**initialize_qpu`** is provided by the Quam machine and is wired as the backend’s `**init_macro**`, so that each QUA program can start with the correct hardware initialization.
2. **Quam ↔ Qiskit Pulse channel mapping**
  They define and expose the mapping between **Quam channels** (e.g. `qubit.xy`, `qubit.z`, `qubit.resonator`, `qubit_pair.coupler`) and **Qiskit Pulse channels** (`DriveChannel`, `ControlChannel`, `MeasureChannel`, etc.). This is stored as the backend’s channel dictionary and is used by:
  - `**get_quam_channel(qiskit_channel)`** — returns the Quam channel for a given Qiskit Pulse channel  
  - `**get_pulse_channel(quam_channel)**` — returns the Qiskit Pulse channel for a given Quam channel
  With this mapping, users can write **Qiskit Pulse schedules** in Qiskit and convert them natively to QUA (e.g. via `schedule_to_qua_macro` or by adding pulse operations to the backend), using the same channel semantics as the rest of the Quam setup.

## Qiskit Primitives on QOP

We provide custom implementations of the standard Qiskit Primitives, `QMEstimatorV2` and `QMSamplerV2`, which are straightforward adaptations of the [standard Qiskit primitives](https://quantum.cloud.ibm.com/docs/en/guides/primitives). They leverage the core capabilities of the Quantum Orchestration Platform to optimize execution through:

1. **Real-time Parameter Adjustment**: The ability to adjust parameter values in real-time and load them asynchronously using **Input Streaming** or **DGX Quantum**.
2. **Real-time Control Flow**: The ability to perform real-time control flow to estimate different expectation values seamlessly across a single compilation of a quantum circuit (specifically for the Estimator primitive).

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

### Primitive options (QMSamplerOptions and QMEstimatorOptions)

Both primitives accept an options object that controls how jobs are run and how parameters are loaded on the OPX.

#### QMSamplerOptions


| Option          | Type                                         | Default        | Description                                                                                                                                                                                                    |
| --------------- | -------------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `default_shots` | `int`                                        | `1024`         | Default number of shots per circuit when not specified in `run()`.                                                                                                                                             |
| `input_type`    | `InputType | None`                           | `None`         | How parameter values are loaded on the OPX: `InputType.INPUT_STREAM`, `InputType.IO1`, `InputType.IO2`, `InputType.DGX_Q`, or `None` (preload at compile time; use only for a small number of parameter sets). |
| `run_options`   | `dict | None`                                | `None`         | Extra options passed through to the backend’s `run()` method.                                                                                                                                                  |
| `meas_level`    | `"classified" | "kerneled" | "avg_kerneled"` | `"classified"` | Measurement level: classified (counts), kerneled (raw IQ per shot), or avg_kerneled (averaged).                                                                                                                |


#### QMEstimatorOptions


| Option              | Type               | Default    | Description                                                                                         |
| ------------------- | ------------------ | ---------- | --------------------------------------------------------------------------------------------------- |
| `default_precision` | `float`            | `0.015625` | Default precision for expectation-value estimation when not specified in `run()` (e.g. 1/√4096).    |
| `abelian_grouping`  | `bool`             | `True`     | Whether to group observables into qubit-wise commuting sets.                                        |
| `input_type`        | `InputType | None` | `None`     | Same as for the Sampler: `INPUT_STREAM`, `IO1`, `IO2`, `DGX_Q`, or `None` for compile-time preload. |
| `run_options`       | `dict | None`      | `None`     | Extra options passed through to the backend’s `run()` method.                                       |


**InputType** (from `qiskit_qm_provider`): `INPUT_STREAM` (real-time input stream), `IO1`, `IO2` (I/O channels), or `DGX_Q` (DGX Quantum communication). Use `None` to bind all parameter values at compile time.

Standalone examples for the Sampler and Estimator are in the [examples](examples/) folder.

## Hybrid QUA and Qiskit Interface

We envision this tool as more than just a Qiskit bridge; it is a new interface to intertwine the power of Qiskit and QUA. You can build QUA programs over many qubits that incorporate the execution of Qiskit quantum circuits as QUA-embedded macros.

### Workflow: Embedding and Processing

When embedding Qiskit circuits into QUA programs, the typical workflow involves two steps using `backend.quantum_circuit_to_qua()` and `get_measurement_outcomes()`:

1. `**backend.quantum_circuit_to_qua(qc, ...)`**: This function compiles the Qiskit circuit into QUA instructions and inserts them into the current QUA program context. It returns a result object.
2. `**get_measurement_outcomes(qc, result, compute_state_int=True)**`: This utility function takes the circuit and the result from the previous step. It returns a dictionary containing all the circuit classical registers names (as you would collect them from `qc` by doing `[creg.name for creg in qc.cregs]`) as keys and the following dictionaries as values:
  - `"value"`: QUA array of boolean variables storing all the measured classical bits included in the `ClassicalRegister` object.
  - `"size"`:  The size (Python integer) relative to the `ClassicalRegister` (i.e., its number of bits).
  - `**state_int**`, a QUA integer variable representing the integer representation formed by all the bits measured in this register. This can be useful for bitpacking.
  - `"stream"`: a stream object that can be retrieved to perform arbitrary saving of the variables obtained by the circuit for this register, and that can be used for buffering in the `stream_processing` segment of the QUA program

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

For scalable error correction workflows, where hybrid classical-quantum computing is essential, we introduce the **Parameter Table**. This module provides a full interface to express parametric programs and seamless communication between a client (or DGX Quantum server) and the QUA program.

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

Class enabling the mapping of parameters to be updated to their corresponding "to-be-declared" QUA variables. It acts as a single entrypoint to update a parameter from both Python and QUA interface.

#### Initialization

```python
ParameterTable(parameters_dict, name=None)
```

- `parameters_dict`: Dictionary `{ "name": (initial_value, qua_type, input_type, direction) }` or list of `Parameter` objects.
- `name`: Optional name for the table.

#### Methods

- `**declare_variables(pause_program=False)**`: QUA Macro to declare all QUA variables associated with the table.
- `**load_input_values(filter_function=None)**`: QUA Macro to load input values from the input stream/IO/DGX Quantum.
- `**push_to_opx(param_dict, job, qm, verbosity)**`: Client function to push values to the OPX.
- `**fetch_from_opx(job, fetching_index, fetching_size)**`: Client function to fetch values from the OPX.
- `**stream_back(reset=False)**`: QUA Macro to stream values back to the client/server.
- `**from_qiskit(qc, input_type, filter_function)**`: Class method to create a table from a Qiskit QuantumCircuit's parameters.

**Note — ParameterVector and OpenQASM 3:** A known limitation of the Qiskit OpenQASM 3 exporter is that `ParameterVector` instances are exported as a series of individual parameters (one per element) rather than as a single array. This provider supports this by creating one parameter per element when building a table with `ParameterTable.from_qiskit`, so behaviour is correct and nothing changes from the user’s perspective; it is only an implementation detail to be aware of.

### `Parameter`

Represents a single parameter mapped to a QUA variable.

- `**assign(value)`**: QUA Macro to assign a value to the parameter's QUA variable.
- `**save_to_stream()**`: QUA Macro to save the current value to its output stream.
- Each Parameter stores a `**var**` attribute that corresponds to the QUA variable associated with the parameter. It can be a QUA int, fixed, bool, or a QUA array of those types.
- We have two special types of Parameters: `QUA2DArray` and `QUAArray`that can be used for multiple indexing as if you were traversing a multi-dimensional array (encoded behind the scens as a single large UQUA array of flattened dimension).

## Compatibility and Custom Calibrations

This provider is compatible with both **Qiskit 1.x** and **Qiskit 2.x**.

### Philosophy: Qiskit embedded in QUA

The provider is built in two layers. The first is the **traditional** one: run Qiskit circuits via `backend.run()` or the primitives (`QMSampler`, `QMEstimator`); the backend compiles circuits to QUA and executes them, with optional real-time parameter and control-flow features. The second layer is an **extended** use of Qiskit: circuits are not only submitted as jobs but can be **embedded inside larger QUA programs** via `quantum_circuit_to_qua`. In that regime, Qiskit is used to define subroutines (circuits and, where applicable, Pulse schedules) that are inlined as QUA macros, with parameters and classical inputs supplied through **ParameterTables**—bound in real time in QUA rather than at Python compile time. That extension is what enables tight integration with real-time QUA processing and hybrid classical–quantum workloads (feedback, streaming, DGX, etc.) while still writing algorithms in familiar Qiskit terms. Custom gates and calibrations (below) are the way to teach the backend new circuit-level or pulse-level operations so that both the Qiskit Target and the OpenQASM3→QUA compiler stay in sync.

### Qiskit 1.x (Pulse Support)

Sticking to Qiskit 1.x enables partial support for **Qiskit Pulse**, allowing custom pulse-level calibrations expressed in Qiskit Pulse to be directly translated into a QUA macro.

### Qiskit 2.x (Qiskit Pulse Deprecation)

We encourage the adoption of Qiskit 2.0. The novel way to express custom calibrations is through `QMInstructionProperties`. This allows you to specify additional gates in the backend target that contain a customized QUA macro.

#### Example: Parametric CNOT Gate with Custom QUA Macro

This example demonstrates how to add a custom parametric gate to the hardware backend using `QMInstructionProperties`.

```python
from qiskit.circuit import Parameter as QiskitParameter, Gate
from qiskit_qm_provider import QMProvider, QMInstructionProperties
from qm.qua import *

# 1. Set up provider and backend
provider = QMProvider("/path/to/quam/state")
backend = provider.get_backend()

# 2. Define an opaque parametric two-qubit gate at the circuit level
theta = QiskitParameter("theta")
cx_cal = Gate("cx_cal", num_qubits=2, params=[theta])  # No logical definition: opaque gate

# (Optional) You may instead provide a logical definition for `cx_cal` so that the transpiler
# can optimize it with other operations; see the Qiskit backend transpiler interface docs:
# https://quantum.cloud.ibm.com/docs/en/api/qiskit/providers#backends-transpiler-interface

# 3. Define the corresponding QUA macro
def qua_macro(theta_val):
    # Here you implement the low-level calibrated pulse sequence
    qubit_pair = backend.get_qubit_pair((0, 1))
    qubit_pair.apply("cz", amplitude_scale=theta_val)

# 4. Register the new instruction in the backend Target
duration = backend.target["cx"][(0, 1)].duration  # Reuse existing CX duration as a template
properties = {
    (0, 1): QMInstructionProperties(
        duration=duration,
        qua_pulse_macro=qua_macro,
    )
}

# This is the essential part of what a helper such as `add_custom_gate` would do:
backend.target.add_instruction(cx_cal, properties=properties)

# 5. Synchronize the internal QUA compiler mapping with the modified Target
backend.update_target()
```

**Important:** whenever you manually modify `backend.target` (e.g. by adding or changing instructions or their
`QMInstructionProperties`), you must call `backend.update_target()` afterwards so that the internal OQ3/QUA
compiler state inside the backend is synchronized with your updated Target before compiling circuits to QUA. The same method can be used if you are manually adding Quam macros into your machine object dynamically, as the sync goes both ways.

Note: When a gate implementation is updated (e.g. the gate was already existing and had an existing pulse level implementation), it always overrides the previously defined implementation when calling the method.

**Why this matters for the Qiskit–QUA embedding:** Adding a custom gate does two things at once: it extends the Qiskit **Target** (so the transpiler knows the gate and can use it), and it registers the corresponding QUA macro in the backend’s operation mapping used by **qm_qasm** when you call `quantum_circuit_to_qua`. Thus the same gate is available both for “standard” Qiskit runs (primitives, `run()`) and for embedding circuits inside larger QUA programs with real-time parameters and hybrid control flow.

## License

This project is licensed under the Apache 2.0 License.

## Attribution & Provenance

This project was initiated and developed by **Arthur Strauss**
as part of his PhD research at the **Centre for Quantum Technologies, National University of Singapore**, in collaboration with **Quantum Machines Ltd.**.

The goal of `qiskit-qm-provider` is to bridge Qiskit-level programming
abstractions (circuits, primitives, and workflows) with the synthesis
and execution of advanced **QUA-based quantum control programs**.

This repository serves as an open-source foundation for future research
and industrial developments in hybrid quantum software stacks.