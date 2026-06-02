# Error-Correction Workflow

Error correction is the **canonical stress test** for hybrid Qiskit-in-QUA design: repeated syndrome measurement, classical decoding, and recovery inside a long-running QUA program while circuits stay authored in Qiskit.

For method signatures, see the [Parameter Table API reference](apidocs/qm_parameter_table.rst) and [Backend API reference](apidocs/qm_backend.rst).

## Purpose

Plain circuit submission breaks down for EC because you need:

- **Many cycles** of measure → decode → recover, not one shot per circuit variant.
- **Streaming classical data** (syndrome integers) between QUA and a host decoder.
- **Stable parameter names** across Qiskit recovery circuits and QUA variables.

[`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) and [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) make that classical-quantum contract explicit. [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst) wires syndrome bits from an embedded circuit into QUA variables in the **same** `with program():` block.

## Why hybrid EC is hard without this toolbox

| Pain point | What goes wrong | How this toolbox helps |
|------------|-----------------|------------------------|
| Circuit explosion | One Qiskit circuit per branch/time step | One syndrome + one recovery circuit, loop in QUA |
| Ad-hoc data plumbing | Mismatched names, shapes, JSON blobs | Central [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) definition |
| Qiskit vs QUA assumptions | Single-run binding vs long-running streams | [`InputType`](apidocs/stubs/qiskit_qm_provider.parameter_table.InputType.rst) streaming + `push_to_opx` / `load_input_values` |

## ParameterTable roles in EC

### Declare the contract once

```python
from qiskit_qm_provider import Parameter, ParameterTable, Direction, InputType

syndrome_data = Parameter(
    "syndrome_data",
    0,
    input_type=InputType.INPUT_STREAM,
    direction=Direction.INCOMING,
)

recovery_vars = ParameterTable.from_qiskit(
    recovery_circuit,
    input_type=InputType.INPUT_STREAM,
)
```

Inside QUA:

```python
recovery_vars.declare_variables()
syndrome_data.declare_variable()
syndrome_data.declare_stream()
```

### Stream syndrome out (same program, right after embedding)

```python
from qiskit_qm_provider.backend.backend_utils import get_measurement_outcomes

syndrome_meas_result = backend.quantum_circuit_to_qua(syndrome_circuit)
syndrome_meas_dict = get_measurement_outcomes(syndrome_circuit, syndrome_meas_result)
state_int_val = syndrome_meas_dict[ancilla_creg.name]["state_int"]

syndrome_data.assign(state_int_val)
syndrome_data.stream_back(reset=True)
```

| Key | Role in EC |
|-----|------------|
| `value` | Per-bit QUA variables from the syndrome measurement |
| `state_int` | Packed integer syndrome (LSB = qubit 0) — usual handle for decoding |
| `size` | Number of syndrome bits |
| `stream` | Stream handle for `stream_processing()` on the host |

See [Backend guide — get_measurement_outcomes](backend.md#get-measurement-outcomes-return-dictionary) for full details.

### Push recovery parameters in

```python
recovery_vars.push_to_opx(param_dict, job, qm, verbosity=0)
```

In QUA:

```python
recovery_vars.load_input_values()
backend.quantum_circuit_to_qua(recovery_circuit, recovery_vars)
```

## End-to-end loop

1. Author syndrome and recovery circuits in Qiskit.
2. Declare [`ParameterTable`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) / [`Parameter`](apidocs/stubs/qiskit_qm_provider.parameter_table.Parameter.rst) for streamed data.
3. Inside `with program():`, call [`quantum_circuit_to_qua`](apidocs/stubs/qiskit_qm_provider.backend.QMBackend.rst) for the syndrome circuit.
4. Call [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst) and `stream_back()` the syndrome integer.
5. On the host, decode and `push_to_opx()` recovery parameters.
6. `load_input_values()` and embed the recovery circuit.

The full QEC program skeleton lives in the [repository README](https://github.com/arthurostrauss/qiskit-qm-provider#error-correction-and-parameter-table).

## Related

- **Guides:** [Backend — embedding](backend.md#embed-in-qua-hybrid), [Parameter Table](parameter_table.md), [Workflows — hybrid](workflows.md#hybrid-qua-qiskit-programs-embedding-circuits-in-qua)
- **API:** [Parameter Table reference](apidocs/qm_parameter_table.rst), [Backend reference](apidocs/qm_backend.rst), [`get_measurement_outcomes`](apidocs/stubs/qiskit_qm_provider.backend.backend_utils.get_measurement_outcomes.rst)
- **Example:** [Error correction in README](https://github.com/arthurostrauss/qiskit-qm-provider#error-correction-and-parameter-table)
