# Parameter Table Submodule: In-Depth Guide

This document provides a detailed presentation of the **Parameter**, **ParameterTable**, **QUAArray**, and **QUA2DArray** classes in `qiskit_qm_provider.parameter_table`. These classes manage dynamic parameters in QUA programs, enabling runtime updates from Python, integration with external I/O (OPX IO1/IO2, input streams), and **QUARC**-backed **OPNIC** packet communication (classical host ↔ OPX) for Reinforcement Learning, Quantum Optimal Control, and similar workflows.

---

## Table of Contents

1. [Overview](#overview)
2. [Parameter](#parameter)
3. [ParameterTable](#parametertable)
4. [QUAArray](#quaarray)
5. [QUA2DArray](#qua2darray)
6. [InputType and Direction](#inputtype-and-direction)
7. [ParameterPool and OPNIC](#parameterpool-and-opnic)
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

### Measurement outputs (dual namespace)

Compiled-circuit measurement fields (`MeasurementRegisterField`, exposed via `comp.outputs`) are **not** runtime `Parameter` objects. They live in a separate weakref-tracked registry on `ParameterPool` and are owned per `QuaCircuitCompilation`.

| Access | Runtime `ParameterTable` | `MeasurementOutcomeTable` |
|--------|--------------------------|---------------------------|
| `table["c"]` / `table.c` | QUA var (inside `with program():`) | QUA var (inside `with program():`) |
| `table.get_parameter("c")` | `Parameter` handle | `MeasurementRegisterField` handle |

QUA variable accessors (`.var`, `__getitem__`, `state_ints`, `streams`, `declare`, `assign`, …) raise outside `with program():`. Use `ParameterPool.lookup_runtime_parameter(name)` for runtime-only name lookup.

See [docs/measurement_outputs.md](../../docs/measurement_outputs.md) for the full locality model.

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
- **direction**: For `InputType.OPNIC` only, from the QUA program's perspective (aligned 1:1 with Quarc): `Direction.INCOMING` (into QUA: classical host → OPX, the `rcv`/`push_to_opx` direction), `Direction.OUTGOING` (out of QUA: OPX → classical host, the `stream_back`/`fetch_from_opx` direction), or `Direction.BOTH`. Required when `input_type == InputType.OPNIC`. **OPNIC is built on QUARC and requires the `quarc` package** — if you need OPNIC and don't have Quarc, contact the Quantum Machines team for access. INPUT_STREAM / IO / plain-stream parameters work without Quarc.
- **units**: Optional units string (e.g. for display).

If `value` is a list or 1D numpy array, the parameter is an array of that length; otherwise it is a scalar. Parameters are **not** globally name-deduplicated: constructing `Parameter("x")` twice yields two distinct objects. (Only `ParameterTable` *names* must be unique, since they become Quarc struct keys.)

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
| `var`          | The QUA variable (after declaration); for an OPNIC struct, returns the field (scalar or array slice). |
| `stream`       | Output stream (if declared) for saving. |
| `input_type`   | `InputType` or `None`. |
| `direction`    | For OPNIC: stream direction from the QUA program's view (`Direction.INCOMING` = into QUA, `Direction.OUTGOING` = out of QUA, `Direction.BOTH`). |
| `incoming_stream_id` / `outgoing_stream_id` | OPNIC Quarc stream ids of the owning `ParameterTable` (OPNIC only). There is no single `stream_id` — the two directions are independent. |
| `struct_type`  | The Quarc `Struct` *type* of the owning OPNIC `ParameterTable` (the former `opnic_struct` attribute was removed; reach the bound handle via the table's `var`). |
| `main_table`   | `ParameterTable` that declared this parameter (OPNIC). |
| `index` / `get_index(table)` | Index in a given `ParameterTable` (useful for `switch_` logic in QUA for deciding which parameter to update). |
| `tables`       | List of tables that contain this parameter. |

### Declaration (inside QUA)

- **`declare(pause_program=False, declare_stream=True)`** *(canonical name)*  
  Declares the QUA variable (and optionally an output stream). For OPNIC, parameters are table-managed and the call delegates automatically through the owning table’s struct handle.  
  *Deprecated alias:* `declare_variable(...)` — identical behaviour, kept for backwards-compatibility.

- **`declare_stream()`**  
  Declares the output stream for this parameter (used for `save_to_stream` / stream processing).

### Assignment (inside QUA)

- **`assign(value, condition=None, value_cond=None)`**  
  Assigns to the parameter’s QUA variable.  
  - **value**: Python scalar, list, another `Parameter`’s `.var`, or a QUA array variable. For arrays, length must match.  
  - **condition** / **value_cond**: Optional QUA boolean and else-value; both must be provided together.  
  For array parameters, assignment can be from a list, a QUA array, or another Parameter; for OPNIC struct, assignment is to the struct field.

### Loading input (inside QUA)

- **`rcv()`** *(canonical name)*  
  Loads one value from the parameter’s input mechanism: advances `INPUT_STREAM` or reads from IO1/IO2. For OPNIC, the call delegates through the owning table’s `recv()`.  
  *Deprecated alias:* `load_input_value()` — identical behaviour, kept for backwards-compatibility.

### Streaming and saving (inside QUA)

- **stream_back(reset=False)**  
  Sends the current value to the client: for non-OPNIC with a stream, saves to stream; for OPNIC with `Direction.OUTGOING` or `Direction.BOTH` (out of QUA), sends the struct to the external stream; for IO1/IO2, sends via IO. If `reset=True`, resets the variable to zero (in appropriate QUA type) after sending.

- **save_to_stream()**  
  Saves the current value to the parameter’s output stream (must be declared).

- **stream_processing(mode="save" | "save_all", buffer="default" | ...)**  
  Defines how the output stream is processed (e.g. `save` vs `save_all`). For arrays, default buffer is `(length,)`; for scalars, no buffer.

### Utilities (inside QUA)

- **clip(min_val, max_val, is_qua_array=False)**  
  Clips the QUA variable to `[min_val, max_val]` (element-wise for arrays). Bounds can be scalars or QUA arrays if `is_qua_array=True`.

### Python-side (outside QUA)

- **`push_to_opx(value=<stored>, job=None, qm=None, verbosity=1, time_out=30)`**  
  Sends `value` from Python to the OPX so that the next `rcv()` call in QUA receives it.  
  Uses input stream, IO1/IO2, or OPNIC depending on `input_type`.  
  **Zero-argument form:** when `value` is omitted, `self.value` (the Python-side stored float set at construction or later) is used. This lets you update the stored value once and call `push_to_opx()` without repeating the argument:
  ```python
  param.value = new_angle
  param.push_to_opx()   # streams new_angle
  ```

- **`fetch_from_opx(job=None, fetching_index=0, fetching_size=1, verbosity=1, time_out=30)`**  
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

- **`declare(pause_program=False, declare_streams=True)`** *(canonical name)*  
  Declares all parameters. For OPNIC, declares the packet struct and external stream(s), and binds each parameter to the packet; for non-OPNIC, calls `declare()` on each parameter. Returns the declared variable(s) (single var or list; for OPNIC, the packet).  
  *Deprecated alias:* `declare_variables(...)` — identical behaviour, kept for backwards-compatibility.

- **`declare_stream()`** *(canonical name)*  
  Declares output streams for all parameters that do not already have one.  
  *Deprecated alias:* `declare_streams()` — identical behaviour.

### Assignment (inside QUA)

- **assign_parameters(values: Dict[str | Parameter, value])**  
  Assigns multiple parameters at once. Keys are names or `Parameter` instances; values are Python literals, QUA expressions, or `Parameter` instances.

- **table[name] = value** or **table.attribute = value**  
  Assigns to the parameter named `name` (or attribute); equivalent to `table[name].assign(value)`.

### Loading and streaming (inside QUA)

- **`rcv(filter_function=None)`** *(canonical name)*  
  For each parameter (optionally filtered), calls `rcv()` on it. For OPNIC INCOMING/BOTH (into QUA), receives one packet into the table’s struct.  
  *Deprecated alias:* `load_input_values(...)` — identical behaviour.

- **`stream_back(reset=False)`**  
  For each parameter (or for OPNIC INCOMING, sends the whole packet) streams values out; see Parameter’s `stream_back`. Optionally resets after send.

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

### Mutating the table (before emission)

- **add_parameters(Parameter \| List[Parameter])**
  Appends parameter(s); indices are assigned automatically. All must have the same
  `input_type` (and direction for OPNIC). Allowed only while
  `table._is_emitted == False` — once the table's struct has been emitted to a module
  (i.e. after `declare()`, or after a Flow-B handle was bound), `add_parameters` raises
  because Quarc's `add_struct` is append-only and cannot retroactively grow a struct's
  field set. Modify the table before declaring it.

- **remove_parameter(name \| Parameter)**
  Removes a parameter and properly detaches it (`Parameter._unset_index`); for
  OPNIC parameters that lose their last attachment to a non-synthetic table, the
  parameter is re-registered in `ParameterPool._pending_standalone_opnic`. Same
  emission lock as `add_parameters`.

- **add_table(ParameterTable \| List[ParameterTable])**
  Merges another table(s) by adding its parameters. Same emission lock as
  `add_parameters`.

### Properties

- **parameters**: List of `Parameter` objects.
- **table** / **parameters_dict**: Name → `Parameter` mapping.
- **variables**: List of QUA variables (after declaration).
- **variables_dict**: Name → QUA variable (after declaration).
- **is_declared**: True if all parameters are declared.
- **input_type**, **direction**: For OPNIC tables.
- **packet**, **struct_type**, **incoming_stream_id** / **outgoing_stream_id**: OPNIC-only.

### Creation from Qiskit

- **ParameterTable.from_qiskit(qc, input_type=None, filter_function=None, name=None)**  
  Builds a `ParameterTable` from a `QuantumCircuit`’s parameters (symbolic and/or classical). Optionally filter parameters and set `input_type` and table name. Returns `None` if no parameters.

### Python-side

- **push_to_opx(param_dict, job=None, qm=None, verbosity=1)**  
  Pushes a dictionary of name → value (or Parameter → value) to the OPX. For OPNIC OUTGOING, all packet fields must be provided and one packet is sent.

- **fetch_from_opx(job=None, fetching_index=0, fetching_size=1, verbosity=1, time_out=30)**  
  Fetches from OPX and returns a dictionary name → value. For OPNIC INCOMING, reads packet(s) from the QUARC stream.

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

### Python-side

- **`push_to_opx(value, job=None, qm=None, verbosity=1, time_out=30)`**
  **value** must be a numpy array or nested list whose shape equals `self.shape`; it is flattened row-major and pushed via the base `Parameter.push_to_opx`.
  `job` may be a `RunningQmJob` or `JobApi`. Prefer `JobApi` — current QUA drives IO through the job interface. `qm` is optional legacy back-compat for older job objects that still routed IO via the machine (same contract as `Parameter`).

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

- **`push_to_opx(value, job=None, qm=None, verbosity=1, time_out=30)`**
  **value** must be a 2D array or list of lists of shape `(n_rows, n_cols)`; it is flattened and pushed via the base Parameter’s `push_to_opx`.
  `job` may be a `RunningQmJob` or `JobApi`. Prefer `JobApi` — current QUA drives IO through the job interface. `qm` is optional legacy back-compat for older job objects that still routed IO via the machine (same contract as `Parameter`).

---

## InputType and Direction

**InputType** (enum) controls how a parameter (or table) gets values from or sends values to the outside world:

| Value            | Description |
|------------------|-------------|
| `INPUT_STREAM`   | QUA input stream; advance with `advance_input_stream`; push from Python via job API. |
| `IO1` / `IO2`    | OPX physical IO channels; program pauses, reads value, assigns to variable; Python uses `set_io_values` when job is paused. |
| `OPNIC`          | OP Network Interface Card: QUARC-backed external stream with a packet struct; direction is given by **Direction**. |

**Direction** (enum, used with `InputType.OPNIC`):

| Value      | Meaning |
|------------|---------|
| `INCOMING` | Into the QUA program: classical host → OPX (QUA receives; the `rcv` / `push_to_opx` direction). Maps to Quarc `INCOMING`. |
| `OUTGOING` | Out of the QUA program: OPX → classical host (QUA sends; the `stream_back` / `fetch_from_opx` direction). Maps to Quarc `OUTGOING`. |
| `BOTH`     | Bidirectional (separate streams for in/out). |

For a **ParameterTable** with `InputType.OPNIC`, every parameter must share the same **direction**.

---

## ParameterPool and OPNIC

**ParameterPool** manages unique IDs and registrations for parameter objects/tables, and owns the single Quarc `BaseModule` slot used by all OPNIC transport.

### Internal state

`ParameterPool` is a class-level singleton (every method is a `classmethod`). It tracks four pieces of state:

| Attribute | Purpose |
|---|---|
| `_counter` | Monotonic int allocator for `_id` on every registered `ParameterTable`. |
| `_registry: dict[int, ParameterTable \| Parameter]` | Main registry. Every `ParameterTable` (regular and synthetic-standalone) appears here. Direct `Parameter` registration is rare — typically only happens through legacy paths. Names are unique across the registry. |
| `_pending_standalone_opnic: list[Parameter]` | Solo OPNIC `Parameter`\s that have not been attached to any `ParameterTable` and have not been promoted to a synthetic single-field table. Populated at `Parameter.__init__` (OPNIC only); cleared by `set_index` (when the parameter joins a regular table) or by promotion. Names are also unique against the registry. |
| `_quarc_module: BaseModule \| None` | The single Quarc module slot. Bound either by `from_quarc_module(my_module)` (Pipeline 1) or lazily by `to_quarc_module()` (Pipeline 2). Once bound, calling either method again with a different module raises. Use `reset()` to clear. |

### Public methods

- **`get_id(obj=None) -> int`** — allocates a new id and (if `obj` given) registers it. Raises on duplicate names across `_registry` ∪ `_pending_standalone_opnic`.
- **`get_obj(id) -> ParameterTable | Parameter`** — registry lookup by id.
- **`get_all_objs() / get_all_ids() / get_all()`** — registry views.
- **`iter_opnic_parameter_tables()`** — every OPNIC `ParameterTable` (regular + synthetic-standalone), sorted by id.
- **`iter_standalone_opnic_parameters()`** — union of *(a)* still-pending solo OPNIC `Parameter`\s and *(b)* `Parameter`\s already promoted into a synthetic single-field table.
- **`has_quarc_module() -> bool`** — whether a Quarc module is currently bound.
- **`quarc_module() -> QiskitQMModule`** — getter; lazily creates a default `QiskitQMModule()` if none is bound. OPNIC structs are not emitted here — they emit at `declare()`.
- **`set_quarc_module(m)`** — explicit setter; raises if a module is already bound.
- **`from_quarc_module(m) -> dict[str, ParameterTable | Parameter]`** — Pipeline 1 entry point (see below). **1-field structs are automatically promoted to a standalone `Parameter`** rather than a `ParameterTable` (see [1-field promotion rule](#1-field-struct-promotion-in-from_quarc_module) below).
- **`to_quarc_module(module=None) -> QiskitQMModule`** — Pipeline 2 entry point (see below). Idempotent; default allocation is `QiskitQMModule()`.
- **`reset()`** — clears `_registry`, `_counter`, `_pending_standalone_opnic`, and `_quarc_module`.

### Two pipelines (mutually exclusive once a module is bound)

The pool's two pipelines reflect *who creates the module*. In both cases OPNIC emission
is **declare-time** — every OPNIC `ParameterTable` emits its struct at `declare()` inside
the QUA program (the single commit point), or arrives already-emitted via a Flow-B handle.
Non-OPNIC registration is likewise **declare-time** (specs captured into `parameter_specs`
when `declare()` runs).

**Pipeline 1 — module-first.** The user has a hand-built `quarc.BaseModule` subclass and wants `ParameterTable` wrappers around its existing structs:

```python
class MyModule(BaseModule):
    def __init__(self):
        super().__init__()
        self.add_struct(MyStruct, QuarcDirection.INCOMING)

m = MyModule()
tables = ParameterPool.from_quarc_module(m)
# tables["my_struct"]._var is the QuaStructHandle for MyStruct.
# m is now the pool's accumulator: subsequent OPNIC ParameterTables
# emit onto m at declare() (inside `with program():`).
```

**Pipeline 2 — parameters-first.** No pre-built module — the user declares `Parameter` / `ParameterTable` objects with `input_type=OPNIC` first, then (optionally) constructs the module. Nothing Quarc-side is built at construction: the Quarc `Struct` *type* and the `add_struct` emission are **both deferred to `declare()`** (the single commit point), so you can keep `add_parameters` / `remove_parameter`-ing a table right up until you declare it:

```python
mu    = Parameter("mu", [0.0]*4, input_type=OPNIC, direction=INCOMING)
sigma = Parameter("sigma", [0.1]*4, input_type=OPNIC, direction=INCOMING)
table = ParameterTable([mu, sigma], name="Policy")
solo  = Parameter("theta", 0.0, input_type=OPNIC, direction=INCOMING)

# pool._quarc_module is None; table is "pending" (_is_emitted=False, _var=None);
# solo is in pool._pending_standalone_opnic.

from qiskit_qm_provider import QiskitQMModule

m = QiskitQMModule()  # binds the pool slot. Emits NOTHING yet — m._structs is empty.

with program():
    table.declare()   # emits "Policy" onto m now (mints stream ids), binds the handle.
    solo.declare()    # promotes solo to a synthetic 1-field table "theta_packet" and
                      # emits it now.
# m._structs == {"Policy": ..., "theta_packet": ...}
```

**`declare()` is the single commit point.** OPNIC emission (`add_struct` + stream-id
minting) happens there and nowhere else — not at `Parameter` / `ParameterTable`
construction, not at `QiskitQMModule()` construction. Consequences:

- Emission order = `declare()` order within the one program → deterministic stream ids.
- A struct constructed but never declared is (correctly) absent from `m._structs`;
  serialize the module *after* the program is built. `to_dict()` warns if un-emitted
  OPNIC tables exist.
- **One program per process for OPNIC**: declaring OPNIC tables under a second, distinct
  `with program():` raises. Call `ParameterPool.reset()` to start a new program.

> **Custom `BaseModule` callers (Pipeline 1).** `ParameterPool.to_quarc_module()`
> allocates a `QiskitQMModule` when the slot is empty. Passing a hand-built module via
> `from_quarc_module(m)` or `set_quarc_module(m)` binds the slot and wraps that module's
> already-emitted structs (Flow B). Newly-constructed OPNIC tables emit onto the bound
> module at their `declare()`, whatever its concrete `BaseModule` subclass.

> **Single-module-per-process invariant.** `QiskitQMModule.__init__` raises
> `RuntimeError` if the pool already has a module bound. To create another module
> in the same process, call `ParameterPool.reset()` first. Test suites typically
> add an autouse fixture for this; see `tests/conftest.py` in `rl_qoc`.

### Locking semantics for standalone OPNIC `Parameter`\s

A solo OPNIC `Parameter` is in one of three states over its life:

1. **Pending** — created with `input_type=OPNIC`, not yet attached, not yet declared. Lives in `_pending_standalone_opnic`. Free to be attached to a `ParameterTable`.
2. **Attached** — `set_index` has placed it inside a multi-or-single-field regular `ParameterTable`. Removed from `_pending_standalone_opnic`. `parameter.is_stand_alone == False`. Field-level OPNIC methods (`declare_variable`, `push_to_opx`, …) raise `RuntimeError("table-managed …")` and direct the caller to use the table's API.
3. **Promoted** — on its **first `declare()`** (or other transport call) while pending, the parameter is wrapped in a synthetic single-field `ParameterTable` (named `"<name>_packet"`) which emits its struct at that `declare()`. `parameter._main_table` / `parameter.opnic_table` point at the synthetic. From this moment the parameter is *locked*: trying to attach it to any other `ParameterTable` raises.

`parameter.is_stand_alone` is `True` for states 1 and 3 (pending or promoted) and `False` for state 2 (attached to a regular table).

> **A standalone OPNIC parameter never declared is absent from the artifact.** Because
> emission is deferred to `declare()`, a parameter that is constructed but never declared
> in a program contributes no struct to `m._structs`. Serialize the module *after* the
> program is built; `to_dict()` warns if un-emitted OPNIC tables remain.

### Stream-id consumption point

`module.add_struct(...)` is **append-only** in Quarc and consumes one (or two, for `BOTH`) stream id from the global `_incoming_ids` / `_outgoing_ids` counters (capped at 1023 each). Under the single-commit-point model, ids are consumed **only**:
- at `declare()` (Flow A) — the table (or a promoted standalone's synthetic table) emits then, minting ids in declare() order; or
- when wrapping a pre-built handle in `from_quarc_module(m)` (Flow B) — no minting, the ids are read back from the handle.

`ParameterPool.reset()` resets these counters, so each fresh program/process starts from a clean, reproducible id sequence. A table/parameter that is never declared consumes no id.

### Pool's bound module — invariants

- One slot, one `BaseModule`. Re-binding raises until `reset()`.
- The pool holds **weak** references to `ParameterTable`s (no leak); `Parameter`s are reached through their tables.
- OPNIC emission happens exclusively at `declare()` (or via a pre-built `_quarc_handle` in Flow B) — never at construction or module binding.
- OPNIC is one-program-per-process: declaring under a second distinct program scope raises.

### Parameter identity (no global dedup)

`Parameter(...)` always constructs a **fresh object** — there is no global name lookup or
dedup. Two parameters may share a name across different tables; identity is the Python
object itself. Per-transport identity:

- **INPUT_STREAM** — the parameter *name* is the wire identifier (the QUA input-stream
  name; uniqueness among input streams in one program is enforced by QUA itself).
- **OPNIC** — the integer stream id minted by Quarc at `declare()` (or read back on
  reconstruction). Names are only struct/field keys.
- **IO1 / IO2** — the global IO register.

Only `ParameterTable` *names* must be unique (they become Quarc struct keys —
`module._structs[name]` would otherwise silently overwrite). `ParameterPool.lookup_runtime_parameter(name)`
remains available for name→`Parameter` lookup across registered tables.

### Cross-process caveat

The `BaseModule` returned by `to_quarc_module()` is a real in-process Quarc module. If the QUA-side and classical-side code execute in the same Python session (typical for `quarc.run()` driven from a single entry point) they share the same module instance and stream ids match by reference. For multi-process deployments, the classical side must rebuild the wrapper tables — typically by calling `from_quarc_module(my_module)` against the same `BaseModule` subclass / serialized representation. Quarc's own dynamic-module workflow (declaring structs at runtime rather than in static source) only works cleanly when both sides share the assembled module.

In QUA: **`declare()`**, **`rcv()`** (for INCOMING tables — into QUA), **`stream_back()`** (for OUTGOING — out of QUA). From Python: **`push_to_opx()`** (INCOMING) or **`fetch_from_opx()`** (OUTGOING).

---

## Quarc hybrid alignment

Authoritative packet layout and stream ids come from **Quarc**. Both pipelines (see ParameterPool section above) end up in the same final state: `table._var` is a `QuaStructHandle`; QUA-side declaration runs through `initialize_in_qua()`; transport flows through `send` / `recv`.

**QUA mapping (same `qm.qua` primitives as `QuaStructHandle`):**

| `ParameterTable` / `Parameter` (OPNIC) | Quarc `QuaStructHandle` |
|----------------------------------------|-------------------------|
| `declare()` (`declare_struct` + `declare_external_stream`) | `initialize_in_qua()` |
| `rcv()` → `receive_from_external_stream` | `recv()` |
| `stream_back()` → `send_to_external_stream` | `send()` |

Directions follow the same **`Direction`** enum as elsewhere in this doc, from the QUA
program's perspective and aligned 1:1 with Quarc: **INCOMING** = into QUA (classical → OPX,
`rcv`/`push_to_opx`), **OUTGOING** = out of QUA (OPX → classical, `stream_back`/`fetch`).

For every OPNIC table currently registered in the pool, call **`ParameterPool.iter_opnic_parameter_tables()`**—sorted by table `_id`. For every standalone OPNIC parameter (pending or promoted), call **`ParameterPool.iter_standalone_opnic_parameters()`**.

### Lifecycle walkthrough — Pipeline 2 (parameters-first)

```python
# Phase 0. Empty pool.
ParameterPool.reset()
# pool._registry = {}, pool._pending_standalone_opnic = [], pool._quarc_module = None

# Phase 1. Declare a multi-field OPNIC ParameterTable.
mu    = Parameter("mu", [0.]*4, input_type=OPNIC, direction=INCOMING)
sigma = Parameter("sigma", [.1]*4, input_type=OPNIC, direction=INCOMING)
# pool._pending_standalone_opnic = [mu, sigma]
table = ParameterTable([mu, sigma], name="Policy")
# table registered (weakly) in the pool; mu and sigma removed from pending (set_index).
# table._struct_type built (cheap), table._is_emitted=False, table._var=None.

# Phase 2. Declare a standalone OPNIC Parameter.
solo = Parameter("theta", 0., input_type=OPNIC, direction=INCOMING)
# pool._pending_standalone_opnic = [solo]; solo is still attachable to any future table.

# Phase 3. Bind a module. This does NOT emit anything — it only binds the pool slot.
from qiskit_qm_provider import QiskitQMModule

m = QiskitQMModule()
# - pool._quarc_module = m; m._structs is still EMPTY.

# Phase 4. Emission happens at declare() inside the program (single commit point).
with program() as prog:
    table.declare()   # add_struct("Policy") now: table._is_emitted=True, _var bound.
    solo.declare()    # promotes solo -> synthetic table "theta_packet", emits it now.
# m._structs == {"Policy": ..., "theta_packet": ...}.
# (Declaring OPNIC tables in a *second* distinct program scope raises — reset() first.)

# Phase 5. Once promoted, attaching ``solo`` to another ParameterTable raises
# (it is locked into its synthetic single-field table).
ParameterTable([solo], name="other")  # ValueError: standalone-locked.
```

### Lifecycle walkthrough — non-OPNIC (declare-time registration)

```python
# Module bound first (or via Pipeline 2 sweep — order doesn't matter for non-OPNIC).
m = QiskitQMModule()

# Construct a non-OPNIC ParameterTable AFTER the module exists.
n_reps   = Parameter("n_reps_var", 0, qua_type="int", input_type=InputType.INPUT_STREAM)
n_shots  = Parameter("n_shots",    0, qua_type="int", input_type=InputType.INPUT_STREAM)
ctx      = ParameterTable([n_reps, n_shots], name="ctx_table")
# parameter_specs is unchanged — non-OPNIC objects do not eagerly register at
# construction; they wait until declare() runs in QUA scope.

# Inside the QUA program, ``declare()`` triggers registration.
with program() as prog:
    ctx.declare()      # appends ctx.to_spec() to m.parameter_specs (idempotent by name)

# Now m.parameter_specs == [
#   {"name": "ctx_table", "input_type": "INPUT_STREAM", "is_table": True,
#    "fields": {"n_reps_var": ..., "n_shots": ...}},
# ]
```

### Lifecycle walkthrough — Pipeline 1 (module-first)

```python
ParameterPool.reset()

# User has a custom quarc.BaseModule subclass with structs already declared.
class MyModule(BaseModule):
    def __init__(self):
        super().__init__()
        self.add_struct(PolicyStruct, QuarcDirection.INCOMING)

# Optionally, the user creates a few OPNIC ParameterTables BEFORE the module call —
# these stay pending.
extra = ParameterTable([Parameter("z", [0.]*2, input_type=OPNIC, direction=OUTGOING)],
                       name="Extra")
# extra._is_emitted=False (pool._quarc_module still None).

# Module-first call:
my_module = MyModule()
wrappers = ParameterPool.from_quarc_module(my_module)
# - wrappers["policy_struct"]._var is the existing QuaStructHandle for PolicyStruct.
# - pool._quarc_module = my_module.
# - extra stays pending (from_quarc_module binds the slot but emits nothing); it will
#   emit onto my_module at its declare() inside the program.

# After this, any new OPNIC ParameterTable / standalone declare() emits onto my_module
# at declare() time — same shared accumulator.
```

---

## Uniform Interface — `Parameter` and `ParameterTable` speak the same language

`Parameter` and `ParameterTable` now expose the same canonical method names for every common operation so that callers never need to branch on the concrete type.

| Concept | Canonical name | Deprecated alias (still works) |
|---|---|---|
| Declare QUA variable(s) | `declare(...)` | `declare_variable()` / `declare_variables()` |
| Receive / load input | `rcv(...)` | `load_input_value()` / `load_input_values()` |
| Declare output stream(s) | `declare_stream()` | `declare_stream()` / `declare_streams()` |
| Zero-reset QUA variable(s) | `reset_qua()` | `reset_var()` / `reset_vars()` |

This means code that holds a `param` variable typed as `Parameter | ParameterTable` (or just "one scalar field vs many") can call `param.declare()` and `param.rcv()` uniformly without an `isinstance` check.

```python
# Before: had to know the concrete type
if isinstance(param, Parameter):
    param.declare_variable()
    ...
    param.load_input_value()
else:
    param.declare_variables()
    ...
    param.load_input_values()

# After: one line, works for both
param.declare()
...
param.rcv()
```

The old singular/plural names are preserved as **deprecated aliases** on both classes; they call through to the same implementation and will continue to work until an explicit deprecation removal pass.

---

## 1-field struct promotion in `from_quarc_module()`

When `ParameterPool.from_quarc_module(module)` deserializes a Quarc module, any struct that contains **exactly one field** is automatically promoted to a standalone `Parameter` instead of a `ParameterTable`. The wrapper table is created and registered as a *synthetic-standalone* table (same mechanism used for parameters promoted on first `declare()` call in Pipeline 2), so all OPNIC transport delegates through it correctly.

**Why this matters:** scalar runtime quantities (e.g. `n_shots`, `circuit_choice_var`, `n_reps_var`) are serialized as 1-field Quarc structs. Without promotion, the deserializing side would receive a 1-element `ParameterTable` and have to manually extract the single `Parameter`, reattach it to a synthetic table, and copy stream IDs. The promotion rule makes this automatic.

```python
# Pipeline 1: module has three structs —
#   PolicyParams  (4 fields: mu[4], sigma[4])
#   RewardParams  (1 field:  reward[1])
#   NShots        (1 field:  n_shots scalar)
tables = ParameterPool.from_quarc_module(module)

# Multi-field structs remain ParameterTable:
assert isinstance(tables["policy_params"], ParameterTable)   # True
assert isinstance(tables["reward_params"], ParameterTable)   # True

# Single-field structs are promoted to Parameter:
assert isinstance(tables["n_shots"], Parameter)              # True
# The Parameter still holds a reference to its owning synthetic table,
# so OPNIC transport (declare, rcv, push_to_opx …) works identically:
tables["n_shots"].declare()      # ok — delegates through opnic_table
tables["n_shots"].rcv()          # ok — delegates through opnic_table._var.recv()
tables["n_shots"].push_to_opx()  # ok — uses self.value if set, or pass a scalar
```

Callers that previously checked `len(table.parameters) == 1` and did manual downconversion can now simply use whatever `from_quarc_module` returns, treating scalars and multi-field tables uniformly through the `declare()` / `rcv()` interface.

---

## JSON serialization — `to_spec()` / `from_spec()`

`Parameter` and `ParameterTable` both expose **`to_spec()` / `from_spec()`** for round-tripping
through plain JSON. This is the channel used by `QiskitQMModule` (see next section) to persist
**non-OPNIC** parameters in `rl_qoc_state.json` alongside the Quarc-native `_structs` channel
that handles OPNIC structs.

### `Parameter.to_spec() -> dict`

Serializes a single parameter to a JSON-friendly dict:

| Key | Always present | Notes |
|---|---|---|
| `name` | yes | |
| `qua_type` | yes | one of `"fixed"`, `"int"`, `"bool"` |
| `is_array` | yes | `True` iff `length > 0` |
| `length` | yes | `0` for scalar, `>=1` for array |
| `input_type` | yes | one of `"OPNIC"`, `"INPUT_STREAM"`, `"IO1"`, `"IO2"`, or `None` |
| `direction` | **OPNIC only** | omitted entirely for non-OPNIC parameters (their `.direction` is undefined) |

```python
from qiskit_qm_provider import Parameter, InputType

p = Parameter("n_reps", 5, qua_type="int", input_type=InputType.INPUT_STREAM)
p.to_spec()
# {'name': 'n_reps', 'qua_type': 'int', 'is_array': False, 'length': 0,
#  'input_type': 'INPUT_STREAM'}
```

### `Parameter.from_spec(spec) -> Parameter`

Reverse of `to_spec()`. The `length` field controls scalar-vs-array:
* `length == 0` → constructs a scalar with `value=None` (caller assigns later)
* `length > 0`  → constructs an array of zeros of that length

```python
spec = {'name': 'n_reps', 'qua_type': 'int', 'is_array': False, 'length': 0,
        'input_type': 'INPUT_STREAM'}
p = Parameter.from_spec(spec)
assert p.input_type == InputType.INPUT_STREAM
assert p.length == 0
```

Pool dedup applies: if a `Parameter` with the same `name` already exists in the pool, the
existing instance is returned (after compatibility checks).

### `ParameterTable.to_spec() -> dict`

```python
{
  "name": "n_reps_table",
  "input_type": "INPUT_STREAM",          # shared by every field
  "is_table": True,
  "fields": {
      "n_reps_var": {"qua_type": "int", "is_array": False, "length": 0},
      "n_shots":    {"qua_type": "int", "is_array": False, "length": 0},
  },
  # "direction": "OUTGOING"      ← present only when input_type=="OPNIC"
}
```

### `ParameterTable.from_spec(spec) -> ParameterTable`

Reconstructs the table; each field becomes a `Parameter` (deduped via the pool by name).

```python
table = ParameterTable.from_spec({
    "name": "ctx_table",
    "input_type": "IO1",
    "is_table": True,
    "fields": {"theta": {"qua_type": "fixed", "length": 0, "is_array": False}},
})
assert table.parameters[0].name == "theta"
assert table.input_type.value == "IO1"
```

### When to use `to_spec()` / `from_spec()`

* You are persisting non-OPNIC `Parameter` / `ParameterTable` shapes to JSON for replay.
* You want a *transport-light* serialization that does **not** carry runtime handles.
* You are inspecting `QiskitQMModule.parameter_specs` — its entries are exactly the
  per-object `to_spec()` outputs collected by `_sweep_preexisting_non_opnic` (at
  module-init time) and by the `declare()`-driven hook on `Parameter` / `ParameterTable`.

For OPNIC parameters, prefer the `_structs` channel (handled automatically by
`QiskitQMModule.to_dict()` / `add_struct()`). `to_spec()` *does* support OPNIC parameters
(it includes the `direction` field), but the canonical OPNIC serialization route is the
Quarc-native struct channel — `to_spec()` is exposed for OPNIC mainly for symmetry and
unit-testing convenience.

---

## `QiskitQMModule` — unified non-OPNIC + OPNIC serialization

### When (not) to use `QiskitQMModule` over a plain `BaseModule`

`QiskitQMModule` is the **default** module type for this interface:
`ParameterPool.to_quarc_module()` and `ParameterPool.quarc_module()` lazily allocate
`QiskitQMModule()` when the slot is empty. Use a plain `quarc.BaseModule` subclass only
when you intentionally own struct layout yourself (Pipeline 1) and pass that instance
via `from_quarc_module(m)` or `set_quarc_module(m)`. Either way, OPNIC structs are emitted
at `declare()` — there is no eager sweep onto any module.

`QiskitQMModule` is required for programs that mix **OPNIC** and non-OPNIC parameter
transports. The two practical regimes:

* **Mixed transports** (typical for RL-style workflows that mix tight-latency
  OPNIC packets with slower asynchronous channels):
  - `INPUT_STREAM` parameters: pushed from Python between shots via `push_to_opx()`,
    fetched back through the `RunningQmJob` / `JobApi`. Ideal for parameters that
    change between shots but not within a shot (sweep angles, Hamiltonian
    coefficients, …).
  - `IO1` / `IO2` parameters: pushed asynchronously through OPX hardware IO
    registers — useful for simple scalar control signals when the OPNIC channel
    is reserved for tight-latency reward / policy traffic.
  - Plain QUA vars: tracked in `parameter_specs` so the classical entrypoint
    can reconstruct them via `from_quarc_module(state, opnic_runtime)` /
    `module.reconstruct_non_opnic()` without a runtime dependency.

  `QiskitQMModule.to_dict()` produces a single JSON artifact with both channels
  populated (`_structs` for OPNIC, `parameter_specs` for non-OPNIC), so a single
  `rl_qoc_state.json` (or equivalent) is enough to round-trip the entire program.

### Behaviour

`qiskit_qm_provider.qiskit_qm_module.QiskitQMModule` is a `quarc.BaseModule` subclass that adds
**non-OPNIC parameter persistence** on top of Quarc's existing OPNIC struct channel
*and* **automatic registration** of every `Parameter` / `ParameterTable` constructed
through the pool. There is no `add_parameter` method anymore — registration is driven
entirely by the pool / module binding lifecycle:

Emission/registration is **deferred to `declare()`** (the single commit point), regardless
of whether the module was constructed before or after the objects:

| Constructed... | Registered/emitted... |
|---|---|
| OPNIC `ParameterTable` (before or after the module) | builds its Quarc `Struct` type at construction; **emits via `add_struct` at `declare()`** inside `with program():`, onto the bound module (a default `QiskitQMModule` is created lazily if none is bound). |
| Standalone OPNIC `Parameter` | pending until first `declare()`, then promoted to a synthetic single-field table that emits at that `declare()`. |
| Non-OPNIC `Parameter` / `ParameterTable` | captured into `m.parameter_specs` at `declare()` time. Pre-existing non-OPNIC objects are also captured by `_sweep_preexisting_non_opnic` at module construction. |

(OPNIC is one-program-per-process: declaring under a second distinct `with program():` raises; `reset()` to start over.)

```python
from qiskit_qm_provider import (
    Direction, InputType, Parameter, ParameterTable, ParameterPool, QiskitQMModule,
)

class MyModule(QiskitQMModule):
    pass

# Pipeline 2: parameters first, then module — a single QiskitQMModule() call wires
# everything up. No add_parameter() / add_struct() calls.
mu    = Parameter("mu",    [0.0]*4, input_type=InputType.OPNIC, direction=Direction.INCOMING)
sigma = Parameter("sigma", [0.1]*4, input_type=InputType.OPNIC, direction=Direction.INCOMING)
policy = ParameterTable([mu, sigma], name="PolicyParams")

n_reps = Parameter("n_reps_var", 5, qua_type="int", input_type=InputType.INPUT_STREAM)

m = MyModule()
# - m._structs["PolicyParams"] populated; m._struct_handles[-1] is the QuaStructHandle.
# - m.parameter_specs already includes n_reps (preexisting non-OPNIC sweep).
# - subsequent `n_reps.declare()` inside a QUA program is idempotent; the spec is
#   re-checked by name and not duplicated.

state = m.to_dict()
# state["_structs"]         → {"PolicyParams": {...}}
# state["parameter_specs"]  → [{"name": "n_reps_var", ...}]
```

### `_sweep_preexisting_opnic()` and `_sweep_preexisting_non_opnic()`

Both run automatically during `QiskitQMModule.__init__`, immediately after the module
binds itself as the pool slot:

* `_sweep_preexisting_opnic` is a **no-op** under the single-commit-point model. OPNIC
  structs emit lazily at `ParameterTable.declare()` (Flow A) — never at module
  construction — so there is nothing to sweep. Pre-existing and future OPNIC tables emit
  onto the bound module when they are declared; pending standalone parameters promote and
  emit at their first `declare()`.
* `_sweep_preexisting_non_opnic` walks the registry for non-OPNIC tables / standalone
  parameters and appends each `obj.to_spec()` to `parameter_specs` (idempotent by
  parameter / table name). Non-OPNIC objects carry no stream ids, so capturing their
  specs at module construction is safe.

### `reconstruct_non_opnic() -> dict[str, Parameter | ParameterTable]`

Walks `self.parameter_specs` and rebuilds the original objects via
`Parameter.from_spec()` / `ParameterTable.from_spec()`.  Keys come from each spec's
`attr_name` (legacy artifacts only) — current artifacts use
`pascal_to_snake_case(spec["name"])` so PascalCase parameter names map cleanly onto
snake_case Python attributes (e.g. on `QMEnvironment.circuit_params`).

```python
m2 = MyModule(**state)                  # rebuilds OPNIC structs + carries parameter_specs
non_opnic = m2.reconstruct_non_opnic()  # {'n_reps_var': Parameter('n_reps_var', ...)}
```

### `iter_all_params() -> Iterator[(kind, name, handle_or_spec)]`

Unified iterator over *both* channels:

```python
for kind, name, payload in m.iter_all_params():
    if kind == "opnic":
        # payload is the QuaStructHandle (post add_struct)
        ...
    else:  # kind == "non_opnic"
        # payload is the parameter_specs dict
        ...
```

### Re-instantiation from JSON — replay vs fresh

`QiskitQMModule.__init__` pops `_structs` from incoming `**data` (Pydantic cannot consume a
private attr) and passes it to `_replay_from_structs_data()`, which calls `add_struct()` once
per entry to rebuild `_struct_handles` deterministically. This works *without* a live runtime
— the handles are pure-Python `QuaStructHandle` objects.

```python
state = json.loads(Path("rl_qoc_state.json").read_text())
m = MyModule(**state)
# m._struct_handles is populated (handles, no runtime endpoints yet).
# m.parameter_specs is the non-OPNIC list — call m.reconstruct_non_opnic() to get live objects.
```

---

## Dual-mode `ParameterPool.from_quarc_module(module_or_dict, opnic_runtime=None)`

The classical-side entry point for rebuilding `ParameterTable` / `Parameter` objects from a
serialized Quarc module. It accepts **either** a live module object **or** a plain dict
(loaded directly from `rl_qoc_state.json`).

### Signature

```python
@classmethod
def from_quarc_module(
    cls,
    module: Union["QiskitQMModule", "BaseModule", Dict[str, Any]],
    opnic_runtime: Optional[Any] = None,
) -> Dict[str, "ParameterTable | Parameter"]:
```

### Mode 1 — module object (quantum / generation side)

```python
from qiskit_qm_provider import (
    Direction, InputType, Parameter, ParameterTable, ParameterPool, QiskitQMModule,
)

# Constructed in Pipeline 2 style: tables / standalone params first, then module.
policy_table = ParameterTable(
    [Parameter("mu", [0.0]*4, input_type=InputType.OPNIC, direction=Direction.INCOMING),
     Parameter("sigma", [0.1]*4, input_type=InputType.OPNIC, direction=Direction.INCOMING)],
    name="PolicyParams",
)
n_reps_var = Parameter("n_reps_var", 5, qua_type="int", input_type=InputType.INPUT_STREAM)

m = QiskitQMModule()      # binds the pool; captures pre-existing non-OPNIC specs (OPNIC emits at declare())

tables = ParameterPool.from_quarc_module(m)   # opnic_runtime=None
# tables == {
#   "policy_params": ParameterTable(_var=<QuaStructHandle>),
#   "n_reps_var":    Parameter(input_type=INPUT_STREAM),
# }
```

* OPNIC entries are read from `m._struct_handles` (already built at `add_struct` time).
  Each `ParameterTable` ends up with `_var = QuaStructHandle`.
* Non-OPNIC entries are reconstructed via `m.reconstruct_non_opnic()`.
* `opnic_runtime` is optional; if provided, callers may bind handles afterwards via
  `bind_quarc_runtime()` (the legacy path).

### Mode 2 — plain dict (classical entrypoint)

This is the path used by `classical.py` after `setup_opnic()`:

```python
import json
from pathlib import Path
import rl_qoc_module_opnic as opnic
from qiskit_qm_provider import ParameterPool

state = json.loads(Path("rl_qoc_state.json").read_text())
runtime = opnic.setup_opnic(init_now=True)

tables = ParameterPool.from_quarc_module(state, opnic_runtime=runtime)
# tables["policy_params"]._var  → runtime.policy_params  (live OPNIC endpoint)
# tables["reward_params"]._var  → runtime.reward_params
# tables["n_reps_var"]          → Parameter(INPUT_STREAM)  (no runtime needed)
```

* `opnic_runtime` is **required** in this mode. For each name in `state["_structs"]`, the pool
  reads `getattr(opnic_runtime, snake_case(struct_name))` and assigns it as `_var` directly —
  there is no separate `bind_quarc_runtime()` step.
* Non-OPNIC entries come from `state.get("parameter_specs", [])` and are rebuilt via
  `ParameterTable.from_spec()` / `Parameter.from_spec()` (no runtime dependency).
* Direction is inferred from the presence of `incoming_stream_spec` / `outgoing_stream_spec`
  on each struct (the same convention Quarc uses internally).

### Choosing between the two modes

| You have... | Use... |
|---|---|
| A live `QiskitQMModule` instance (e.g. just built via `RLQoCModule.from_env_and_agent`) | Mode 1 |
| A plain dict from JSON, plus a live OPNIC runtime | Mode 2 |
| A plain dict but **no** runtime (e.g. unit tests, dry-run shape inspection) | Mode 1 — pass the dict to `MyModule(**state)` first to materialize handles, then call `from_quarc_module(m)` |

---

## Usage in QUA Programs

1. **Declare**  
   Call **`param.declare()`** at the start of the program (works for both `Parameter` and `ParameterTable`). Use `declare_streams=True` if you want output streams for saving.

2. **Load input**  
   Where the program should read new values, call **`param.rcv()`** (works for both types; an optional `filter_function` is supported by `ParameterTable`). For OPNIC INCOMING (into QUA), this receives one struct packet.

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
  For OPNIC OUTGOING tables, the dict must contain all parameters of the packet.

- **fetch_from_opx**  
  Parameter: `value = param.fetch_from_opx(job=..., fetching_index=..., fetching_size=..., ...)`.  
  Table: `param_dict = table.fetch_from_opx(...)` returns `{parameter_name: value}`.  
  **fetching_index** and **fetching_size** are used to navigate data saved with **save_all** in stream processing (e.g. iterative workloads); the user must track indices. OPNIC INCOMING reads one packet per call (or multiple if you use fetching_size and the OPNIC API accordingly).

These calls must match the QUA side: each **load_input_value()** / **load_input_values()** is paired with a **push_to_opx** from Python, and each **stream_back()** / stream save is paired with **fetch_from_opx** or result processing on the Python side.

---

This submodule provides a structured way to handle scalar and array parameters, 1D/2D/N-D views, and external I/O (streams, IO, QUARC-backed OPNIC) in QUA programs while keeping a clear contract between the QUA program and the Python client.
