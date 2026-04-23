# Parameter Table Submodule: In-Depth Guide

This document provides a detailed presentation of the **Parameter**, **ParameterTable**, **QUAArray**, and **QUA2DArray** classes in `qiskit_qm_provider.parameter_table`. These classes manage dynamic parameters in QUA programs, enabling runtime updates from Python, integration with external I/O (OPX IO1/IO2, input streams), and OPNIC (OPNIC) communication for Reinforcement Learning, Quantum Optimal Control, and similar workflows.

---

## Table of Contents

1. [Overview](#overview)
2. [Parameter](#parameter)
3. [ParameterTable](#parametertable)
4. [QUAArray](#quaarray)
5. [QUA2DArray](#qua2darray)
6. [InputType and Direction](#inputtype-and-direction)
7. [ParameterPool and OPNIC](#parameterpool-and-dgx-quantum)
8. [Quarc hybrid alignment](#quarc-hybrid-alignment)
9. [Usage in QUA Programs](#usage-in-qua-programs)
10. [Python-Side Interaction](#python-side-interaction)

---

## Overview

- **Parameter**: A single named QUA variable (scalar or 1D array) with optional input/output stream configuration. It maps a Python-side value to a QUA variable that can be declared, assigned, and streamed within a QUA program.
- **ParameterTable**: A named collection of `Parameter` objects sharing the same **input type** (and for OPNIC, the same **direction**). It provides bulk declaration, assignment, loading, and streaming, and for `InputType.OPNIC` it forms a single OPNIC packet.
- **QUAArray**: A **Parameter** subclass that represents an **N-dimensional** logical view over one underlying 1D QUA array. It supports arbitrary shapes, multi-dimensional indexing, slicing, and sub-array views (`_QUAArrayView`).
- **QUA2DArray**: A **Parameter** subclass that represents a **2D** view over one 1D QUA array (`n_rows * n_cols` elements). It offers 2D indexing, row/column slicing, row views (`_QUA2DRow`), and row-wise or element-wise assignment.

All of these can be used inside `with program():` blocks and interoperate with `ParameterTable` (e.g., you can put `Parameter`, `QUAArray`, and `QUA2DArray` instances in a table).

---

## Parameter

The **Parameter** class maps one logical parameter (name + optional units) to a single QUA variable—either a scalar or a 1D QUA array—with type inference, optional input/output configuration, and methods for declaration, assignment, streaming, and Python ↔ OPX communication.

### Initialization

```python
Parameter(
    name: str,
    value: Optional[Union[int, float, List, np.ndarray]] = None,
    qua_type: Optional[Union[str, type]] = None,
    input_type: Optional[InputType | Literal["OPNIC", "INPUT_STREAM", "IO1", "IO2"]] = None,
    direction: Optional[Direction | Literal["INCOMING", "OUTGOING", "BOTH"]] = None,
    units: str = "",
)
```

- **name**: Identifier for the parameter (used in QUA and for table access).
- **value**: Initial value. Can be a scalar (`int`, `float`, `bool`) or a 1D list/array; if omitted, type/length must be set via `qua_type` and (for arrays) by using a subclass like `QUAArray`/`QUA2DArray` that sets length internally.
- **qua_type**: QUA type: `int`, `fixed`, or `bool`. If `None`, inferred from `value` (float → `fixed`, int → `int`, bool → `bool`; for lists/arrays, element type is used).
- **input_type**: How this parameter is fed in or out: `InputType.INPUT_STREAM`, `InputType.IO1`, `InputType.IO2`, or `InputType.OPNIC`. `None` means no external stream (only in-program use and optional QUA stream for saving).
- **direction**: For `InputType.OPNIC` only: `Direction.INCOMING` (OPX → DGX), `Direction.OUTGOING` (DGX → OPX), or `Direction.BOTH`. Required when `input_type == InputType.OPNIC`.
- **units**: Optional units string (e.g. for display).

If `value` is a list or 1D numpy array, the parameter is an array of that length; otherwise it is a scalar. Duplicate names in the same `ParameterPool` (e.g. same name in another table) trigger a warning and reuse of the existing parameter.

### Type inference and QUA types

- **set_type(qua_type)** and **infer_type(value)** (module-level helpers): `set_type` maps Python/string type to QUA type (`fixed`, `int`, `bool`); `infer_type` infers from a scalar or 1D list/array (all elements must share the same type).

### Properties

| Property       | Description |
|----------------|-------------|
| `name`         | Parameter name. |
| `value`        | Initial/value attribute (Python side). |
| `type`         | QUA type (`int`, `fixed`, `bool`). |
| `length`       | Number of elements (0 for scalar). |
| `is_array`     | `True` if `length > 0`. |
| `is_declared`  | Whether the QUA variable has been declared. |
| `var`          | The QUA variable (after declaration); for DGX struct, returns the field (scalar or array slice). |
| `stream`       | Output stream (if declared) for saving. |
| `input_type`   | `InputType` or `None`. |
| `direction`    | For OPNIC: Stream direction (`Direction.OUTGOING`, `Direction.INCOMING`, `Direction.BOTH`). |
| `stream_id`    | OPNIC stream ID of the owning `ParameterTable` (OPNIC only). |
| `opnic_struct`   | Struct handle for the owning OPNIC `ParameterTable`. |
| `main_table`   | `ParameterTable` that declared this parameter (OPNIC). |
| `index` / `get_index(table)` | Index in a given `ParameterTable` (useful for `switch_` logic in QUA for deciding which parameter to update). |
| `tables`       | List of tables that contain this parameter. |

### Declaration (inside QUA)

- **declare_variable(pause_program=False, declare_stream=True)**  
  Declares the QUA variable (and optionally an output stream). For OPNIC, parameters are table-managed and must be declared via `ParameterTable.declare_variables()` from Quarc-built tables.

- **declare_stream()**  
  Declares the output stream for this parameter (used for `save_to_stream` / stream processing).

### Assignment (inside QUA)

- **assign(value, condition=None, value_cond=None)**  
  Assigns to the parameter’s QUA variable.  
  - **value**: Python scalar, list, another `Parameter`’s `.var`, or a QUA array variable. For arrays, length must match.  
  - **condition** / **value_cond**: Optional QUA boolean and else-value; both must be provided together.  
  For array parameters, assignment can be from a list, a QUA array, or another Parameter; for OPNIC struct, assignment is to the struct field.

### Loading input (inside QUA)

- **load_input_value()**  
  Loads one value from the parameter’s input mechanism: advances `INPUT_STREAM` or reads from IO1/IO2. For OPNIC, field-level loading is table-managed via `ParameterTable.load_input_values()`.

### Streaming and saving (inside QUA)

- **stream_back(reset=False)**  
  Sends the current value to the client: for non-OPNIC with a stream, saves to stream; for OPNIC with `Direction.INCOMING` or `Direction.BOTH`, sends the struct to the external stream; for IO1/IO2, sends via IO. If `reset=True`, resets the variable to zero (in appropriate QUA type) after sending.

- **save_to_stream()**  
  Saves the current value to the parameter’s output stream (must be declared).

- **stream_processing(mode="save" | "save_all", buffer="default" | ...)**  
  Defines how the output stream is processed (e.g. `save` vs `save_all`). For arrays, default buffer is `(length,)`; for scalars, no buffer.

### Utilities (inside QUA)

- **clip(min_val, max_val, is_qua_array=False)**  
  Clips the QUA variable to `[min_val, max_val]` (element-wise for arrays). Bounds can be scalars or QUA arrays if `is_qua_array=True`.

### Python-side (outside QUA)

- **push_to_opx(value, job=None, qm=None, verbosity=1, time_out=30)**  
  Sends `value` from Python to the OPX so that the next `load_input_value()` in QUA receives it. Uses input stream, IO1/IO2, or OPNIC OPNIC depending on `input_type`.

- **fetch_from_opx(job=None, fetching_index=0, fetching_size=1, verbosity=1, time_out=30)**  
  Retrieves value(s) from the OPX (IOs, stream processing if `InputType.INPUT_STREAM` or OPNIC packet) and returns them (scalar or array).

---

## ParameterTable

A **ParameterTable** groups multiple **Parameter** instances under one name and ensures they share the same **input_type** (and for OPNIC, the same **direction**). It is the main interface for declaring, assigning, loading, and streaming many parameters at once, and for OPNIC it defines a single OPNIC packet.

### Initialization

**From a dictionary**

```python
parameters_dict: Dict[str, Union[
    Tuple[value, qua_type?, input_type?, direction?],  # up to 4 elements
    value  # scalar or list/array
]]
```

- **value**: Initial value (scalar or 1D list/array).
- **qua_type**: Optional; inferred from value if omitted.
- **input_type**: Optional; if present, all parameters in the table must use the same one.
- **direction**: Required when `input_type == InputType.OPNIC`; must be the same for all.

**From a list of Parameter objects (Recommended)**

```python
ParameterTable([param1, param2, ...], name="optional_name")
```

Input type and direction are taken from the parameters; they must all match. The table assigns each parameter an index (0, 1, …) used e.g. for `switch_` in QUA.

**name**: Optional table name; if not set, a unique default is generated.

### OPNIC packet (InputType.OPNIC)

When `input_type == InputType.OPNIC`, the table builds a QUA struct (packet) whose fields are the parameters. All parameters in the table are then represented by that single struct in QUA; declaration uses `declare_struct` and external streams use this packet type.

### Declaration and streams (inside QUA)

- **declare_variables(pause_program=False, declare_streams=True)**  
  Declares all parameters. For DGX, declares the packet struct and external stream(s), and binds each parameter to the packet; for non-DGX, calls `declare_variable()` on each parameter. Returns the declared variable(s) (single var or list; for DGX, the packet).

- **declare_streams()**  
  Declares output streams for all parameters that do not already have one.

### Assignment (inside QUA)

- **assign_parameters(values: Dict[str | Parameter, value])**  
  Assigns multiple parameters at once. Keys are names or `Parameter` instances; values are Python literals, QUA expressions, or `Parameter` instances.

- **table[name] = value** or **table.attribute = value**  
  Assigns to the parameter named `name` (or attribute); equivalent to `table[name].assign(value)`.

### Loading and streaming (inside QUA)

- **load_input_values(filter_function=None)**  
  For each parameter (optionally filtered), calls `load_input_value()`. For DGX OUTGOING/BOTH, receives one packet into the table’s struct.

- **stream_back(reset=False)**  
  For each parameter (or for DGX INCOMING, sends the whole packet) streams values out; see Parameter’s `stream_back`. Optionally resets after send.

- **save_to_stream(reset=False)**  
  Calls `save_to_stream()` on each parameter. Optionally resets after saving.

- **stream_processing(mode="save" | "save_all", buffering="default" | Dict)**  
  Configures stream processing for all parameters; `buffering` can map parameter name/object to buffer size. By default, 
  buffering for each Parameter in the table will result in a buffering corresponding to the length of the QUA array (if Parameter refers to an array) or no buffering if single variable.

### Accessors

| Method / usage            | Description |
|---------------------------|-------------|
| **get_parameter(name \| index \| Parameter)** | Returns the `Parameter` object. |
| **get_variable(name \| index \| Parameter)** | Returns the QUA variable (after declaration). |
| **get_index(parameter_name \| Parameter)**   | Returns the parameter’s index in this table. |
| **get_type(parameter)**   | Returns the QUA type of the parameter. |
| **has_parameter(name \| index \| Parameter)** | Whether the table contains that parameter. |
| **table[name]** or **table[index]** (after declare) | Returns the QUA variable. |
| **table.name** (attribute) | Same as `table["name"]` for existing parameter names. |

### Mutating the table (before declaration)

- **add_parameters(Parameter \| List[Parameter])**  
  Appends parameter(s); indices are assigned automatically. All must have the same `input_type` (and direction for OPNIC).

- **remove_parameter(name \| Parameter)**  
  Removes a parameter.

- **add_table(ParameterTable \| List[ParameterTable])**  
  Merges another table(s) by adding its parameters.

### Properties

- **parameters**: List of `Parameter` objects.
- **table** / **parameters_dict**: Name → `Parameter` mapping.
- **variables**: List of QUA variables (after declaration).
- **variables_dict**: Name → QUA variable (after declaration).
- **is_declared**: True if all parameters are declared.
- **input_type**, **direction**: For DGX tables.
- **packet**, **opnic_struct**, **stream_id**: DGX-only.

### Creation from Qiskit

- **ParameterTable.from_qiskit(qc, input_type=None, filter_function=None, name=None)**  
  Builds a `ParameterTable` from a `QuantumCircuit`’s parameters (symbolic and/or classical). Optionally filter parameters and set `input_type` and table name. Returns `None` if no parameters.

### Python-side

- **push_to_opx(param_dict, job=None, qm=None, verbosity=1)**  
  Pushes a dictionary of name → value (or Parameter → value) to the OPX. For DGX OUTGOING, all packet fields must be provided and one packet is sent.

- **fetch_from_opx(job=None, fetching_index=0, fetching_size=1, verbosity=1, time_out=30)**  
  Fetches from OPX and returns a dictionary name → value. For DGX INCOMING, reads packet(s) from OPNIC.

---

## QUAArray

**QUAArray** extends **Parameter** to represent an **N-dimensional** logical array stored as one flat 1D QUA array. It supports arbitrary shapes, strides-based indexing, slicing (returning lists or sub-array views), and assignment of elements or full arrays.

### Initialization

```python
QUAArray(
    name: str,
    shape_or_value: Union[Tuple[int, ...], List[Any], np.ndarray],
    qua_type: Optional[Union[str, type]] = None,
    input_type=None,
    direction=None,
    units: str = "",
)
```

- **shape_or_value**:
  - **Tuple of integers** (e.g. `(3, 4, 5)`): Interpreted as **shape**; backing 1D array is allocated with `prod(shape)` elements (initialized to 0).
  - **List or ndarray**: Interpreted as **initial values**; shape is taken from the array’s shape and the flattened list is passed to the base `Parameter` as the initial value.

- **qua_type**, **input_type**, **direction**, **units**: Same semantics as `Parameter`.

Internally, the class computes strides from the shape so that a multi-index `(i, j, k, ...)` maps to flat index `sum(i * stride_i, ...)`.

### Indexing and slicing

- **arr[i, j, k, ...]**  
  If the number of indices equals the number of dimensions, returns the **single QUA variable** at that element (using `_flat_index`).

- **arr[i, ...]** (fewer indices than dimensions)  
  Returns a **_QUAArrayView** on the remaining dimensions, so you can continue indexing: `arr[i][j, k]`, etc.

- **arr[i, :, j]** (slices in any dimension)  
  Slices are expanded in Python: the first slice is turned into a range, and the result is a **list** of variables or views (one per slice index). So `arr[i, :, j]` returns a list of elements for that row.

### Assignment

- **assign(value)**  
  Single argument: assigns to the **whole** array (delegates to base `Parameter.assign` with the flattened value).

- **assign(indices, val)** or **assign((i, j, ...), val)**  
  Two arguments: assign **one element** at the given full index tuple. For partial indices (sub-array), use a view: `arr[i][j, k].assign(...)` or index then assign.

### _QUAArrayView

Slicing with fewer than N indices returns an **_QUAArrayView** that represents a sub-array:

- **view[key]**  
  Further indexing is forwarded to the parent with combined indices.

- **view.assign(val)**  
  Assigns to the view. Fully implemented for **1D views** (e.g. one row): `val` can be a list, numpy 1D array, or QUA array of the same length. Higher-dimensional view assignment may raise `NotImplementedError`.

### Use in ParameterTable

You can put `QUAArray` instances in a `ParameterTable` like any other `Parameter`. Declaration and streaming are inherited; indexing and assignment use the QUAArray API above.

---

## QUA2DArray

**QUA2DArray** is a **Parameter** subclass that provides a **2D** view over one 1D QUA array of length `n_rows * n_cols`. It supports 2D indexing, row/column slicing, row-wise or element-wise assignment, and custom stream processing for 2D data.

### Initialization

```python
QUA2DArray(
    name: str,
    n_rows_or_value: int | List[List[Number]] | np.ndarray,
    n_cols: Optional[int] = None,
    qua_type: Optional[Union[str, type]] = None,
    input_type=None,
    direction=None,
    units: str = "",
)
```

- **n_rows_or_value**:
  - **int**: Number of rows; **n_cols** must be given. Backing array is `n_rows * n_cols` zeros.
  - **2D list of lists** or **2D numpy array**: Shape is taken from the value; backing 1D array is the flattened matrix.

- **n_cols**: Required when first argument is an integer (number of rows).

### Indexing

- **arr[i, j]**  
  Returns the QUA variable at row `i`, column `j` (flat index `i * n_cols + j`).

- **arr[i]**  
  Returns a **_QUA2DRow** proxy so that **arr[i][j]** and **arr[i][slice]** work:
  - **arr[i][j]** → same as **arr[i, j]**.
  - **arr[i][:]** → list of QUA variables for row `i`.

- **arr[i, :]**  
  Row slice: list of variables for row `i` (length `n_cols`).

- **arr[:, j]**  
  Column slice: list of variables for column `j` (length `n_rows`).

- **arr[:, :]**  
  List of lists (all elements).

- **arr[slice]** (e.g. `arr[1:3]`)  
  Returns a list of `_QUA2DRow` objects for those row indices.

### Assignment

- **assign(row, col, val)**  
  Assigns scalar **val** to element `(row, col)`.

- **assign(row, sequence)**  
  Assigns an **entire row**: **sequence** can be a Python list, 1D numpy array, or a QUA array variable of length `n_cols`. Row index is 0-based.

Example:

```python
# One element
arr.assign(0, 0, 1.0)
# Full row from list
arr.assign(0, [1.0, 0.0, 1.0, 0.0])
# Full row from QUA array
arr.assign(0, other_qua_array)
```

### _QUA2DRow

- **row_view[j]** or **row_view[:]**  
  Same as `parent[row, j]` or list for that row.

- **row_view.assign(col_or_vals, val=None)**  
  Delegates to the parent’s `assign(row, col_or_vals, val)`.

### Stream processing

- **stream_processing(mode="save" | "save_all", buffer=None)**  
  For 2D arrays, default buffer is `(n_rows, n_cols)`. You can pass a custom buffer (e.g. `(k, n_cols)` for the last k rows). `save` / `save_all` control whether only the last row or all rows are saved.

### Python-side

- **push_to_opx(value, job, verbosity=1, time_out=30)**  
  **value** must be a 2D array or list of lists of shape `(n_rows, n_cols)`; it is flattened and pushed via the base Parameter’s `push_to_opx`.

---

## InputType and Direction

**InputType** (enum) controls how a parameter (or table) gets values from or sends values to the outside world:

| Value            | Description |
|------------------|-------------|
| `INPUT_STREAM`   | QUA input stream; advance with `advance_input_stream`; push from Python via job API. |
| `IO1` / `IO2`    | OPX physical IO channels; program pauses, reads value, assigns to variable; Python uses `set_io_values` when job is paused. |
| `OPNIC`          | OPNIC (OPNIC): external stream with a packet struct; direction is given by **Direction**. |

**Direction** (enum, used with `InputType.OPNIC`):

| Value      | Meaning |
|------------|---------|
| `INCOMING` | OPX → DGX (OPX sends data to the server). |
| `OUTGOING` | DGX → OPX (server sends parameters to the OPX). |
| `BOTH`     | Bidirectional (separate streams for in/out). |

For a **ParameterTable** with `InputType.OPNIC`, every parameter must share the same **direction**.

---

## ParameterPool and OPNIC

**ParameterPool** manages unique IDs and registrations for parameter objects/tables. In Quarc-backed OPNIC flows, stream transport is table-centric and comes from Quarc struct handles.

- **get_id(obj)**  
  Returns a new unique ID and optionally registers `obj`.

- **get_obj(id)**  
  Returns the object registered for that ID.

- **get_all_objs()**  
  Returns all registered objects.

Typical DGX workflow:

1. Define parameters/tables with `input_type=InputType.OPNIC` and the correct **direction** (they register with the pool).
2. Build OPNIC tables from your module with **`ParameterPool.from_quarc_module(module)`**.
3. In QUA: **declare_variables()**, **load_input_values()** (for OUTGOING tables), **stream_back()** (for INCOMING).
4. From Python: **push_to_opx()** (OUTGOING) or **fetch_from_opx()** (INCOMING).

---

## Quarc hybrid alignment

Authoritative packet layout and stream ids come from **Quarc** (`quarc build` → generated `module.json`). Build `ParameterTable` instances with :meth:`ParameterPool.from_quarc_module` after constructing your `quarc.BaseModule` subclass (e.g. `rl_qoc.qua.quarc.RLQoCModule`). Each table’s ``_var`` is the **`QuaStructHandle`**; Python **push_to_opx** / **fetch_from_opx** delegate to **send** / **recv** on that handle (or the pybind runtime endpoint after rebinding on the classical host).

**QUA mapping (same `qm.qua` primitives as `QuaStructHandle`):**

| `ParameterTable` (OPNIC) | Quarc `QuaStructHandle` |
|--------------------------|-------------------------|
| `declare_variables()` (`declare_struct` + `declare_external_stream`) | `initialize_in_qua()` |
| `load_input_values()` → `receive_from_external_stream` | `recv()` |
| `stream_back()` → `send_to_external_stream` | `send()` |

Directions follow the same **`Direction`** enum as elsewhere in this doc (classical-centric: **OUTGOING** = classical → OPX, **INCOMING** = OPX → classical).

For every OPNIC table currently registered in the pool, call **`ParameterPool.iter_opnic_parameter_tables()`**—sorted by table `_id`.

---

## Usage in QUA Programs

1. **Declare**  
   Call **ParameterTable.declare_variables()** at the start of the program. Use **declare_streams=True** if you want to save results to streams.

2. **Load input**  
   Where the program should read new values, call **load_input_values()** on the table (or **load_input_value()** on a parameter). For DGX OUTGOING, this receives one packet.

3. **Assign**  
   Use **assign** / **assign_parameters** or **table[name] = value** to set variables inside the program. For **QUAArray** / **QUA2DArray**, use their indexing and **assign** methods.

4. **Stream back / save**  
   Use **stream_back()** to send data to the client (and optionally reset). Use **save_to_stream()** and **stream_processing()** for standard QUA result streams.

5. **Switch by index**  
   Use **parameter.get_index(table)** or table’s **get_index(parameter)** to get the parameter index and drive **switch_** logic in QUA.

---

## Python-Side Interaction

- **push_to_opx**  
  Parameter: `param.push_to_opx(value, job=..., qm=..., ...)`.  
  Table: `table.push_to_opx({name: value, ...}, job=..., qm=..., ...)`.  
  For DGX OUTGOING tables, the dict must contain all parameters of the packet.

- **fetch_from_opx**  
  Parameter: `value = param.fetch_from_opx(job=..., fetching_index=..., fetching_size=..., ...)`.  
  Table: `param_dict = table.fetch_from_opx(...)` returns `{parameter_name: value}`.  
  **fetching_index** and **fetching_size** are used to navigate data saved with **save_all** in stream processing (e.g. iterative workloads); the user must track indices. DGX INCOMING reads one packet per call (or multiple if you use fetching_size and the OPNIC API accordingly).

These calls must match the QUA side: each **load_input_value()** / **load_input_values()** is paired with a **push_to_opx** from Python, and each **stream_back()** / stream save is paired with **fetch_from_opx** or result processing on the Python side.

---

This submodule provides a structured way to handle scalar and array parameters, 1D/2D/N-D views, and external I/O (streams, IO, DGX) in QUA programs while keeping a clear contract between the QUA program and the Python client.
