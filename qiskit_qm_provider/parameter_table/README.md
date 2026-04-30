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
- **`quarc_module() -> BaseModule`** — getter; lazily creates a default `quarc.BaseModule()` if none is bound.
- **`set_quarc_module(m)`** — explicit setter; raises if a module is already bound.
- **`from_quarc_module(m) -> dict[str, ParameterTable]`** — Pipeline 1 entry point (see below).
- **`to_quarc_module(module=None) -> BaseModule`** — Pipeline 2 entry point (see below). Idempotent.
- **`reset()`** — clears `_registry`, `_counter`, `_pending_standalone_opnic`, and `_quarc_module`.

### Two pipelines (mutually exclusive once a module is bound)

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
# eagerly emit onto m at construction.
```

**Pipeline 2 — parameters-first.** No pre-built module — the user just declares `Parameter` / `ParameterTable` objects with `input_type=OPNIC`. Tables build the Quarc `Struct` *type* eagerly (cheap; no stream ids consumed) but defer `add_struct` until the user explicitly summons the module:

```python
mu    = Parameter("mu", [0.0]*4, input_type=OPNIC, direction=OUTGOING)
sigma = Parameter("sigma", [0.1]*4, input_type=OPNIC, direction=OUTGOING)
table = ParameterTable([mu, sigma], name="Policy")
solo  = Parameter("theta", 0.0, input_type=OPNIC, direction=OUTGOING)

# At this point: pool._quarc_module is None; table is "pending" (_is_emitted=False);
# solo is in pool._pending_standalone_opnic.

m = ParameterPool.to_quarc_module()
# m is a fresh BaseModule(); table is now emitted onto m and table._var is bound.
# solo is still pending — to_quarc_module() does NOT promote standalone parameters.
# It only sweeps tables. Standalone parameters are promoted lazily on first
# `solo.declare_variable()` (or any transport call), at which point a synthetic
# single-field ParameterTable named "theta" is constructed and added to m.
```

After either pipeline binds the module, every newly-constructed OPNIC `ParameterTable` eagerly emits onto that module — no extra ceremony needed.

### Locking semantics for standalone OPNIC `Parameter`\s

A solo OPNIC `Parameter` is in one of three states over its life:

1. **Pending** — created with `input_type=OPNIC`, not yet attached, not yet declared. Lives in `_pending_standalone_opnic`. Free to be attached to a `ParameterTable`.
2. **Attached** — `set_index` has placed it inside a multi-or-single-field regular `ParameterTable`. Removed from `_pending_standalone_opnic`. `parameter.is_stand_alone == False`. Field-level OPNIC methods (`declare_variable`, `push_to_opx`, …) raise `RuntimeError("table-managed …")` and direct the caller to use the table's API.
3. **Promoted** — `Parameter.declare_variable` (or any other transport call) was invoked while still pending. The pool builds a synthetic single-field `ParameterTable` named after the parameter, force-emits its struct (lazily creating a default `BaseModule` if none is bound yet), and points `parameter._main_table` at the synthetic. From this moment the parameter is *locked*: trying to attach it to any other `ParameterTable` raises.

`parameter.is_stand_alone` is `True` for states 1 and 3 (pending or promoted) and `False` for state 2 (attached to a regular table).

### Stream-id consumption point

`module.add_struct(...)` is **append-only** in Quarc and consumes one (or two, for `BOTH`) stream id from the global `_incoming_ids` / `_outgoing_ids` counters (capped at 1023 each). All emission paths in the pool consume ids at `add_struct` time:
- Pipeline 1's `from_quarc_module(m)` — for any pending OPNIC table swept onto `m`.
- Pipeline 2's `to_quarc_module()` — for every pending OPNIC table swept onto the freshly-built (or previously-bound) module.
- Eager emission triggered when an OPNIC `ParameterTable` is constructed *after* the module is already bound.
- Synthetic-table promotion when a standalone `Parameter` is first declared.

A pending table that is never emitted (e.g. user constructed it but never calls `to_quarc_module()` and never enters QUA scope) does *not* consume any stream id. A pending standalone `Parameter` that is never declared is similarly inert.

### Pool's accumulating module — invariants

- One slot, one `BaseModule`. Re-binding raises until `reset()`.
- Once bound, every OPNIC `ParameterTable.__init__` either (a) eagerly calls `add_struct` on it, (b) accepts an externally-supplied `_quarc_handle` (only used internally by `from_quarc_module`), or (c) — for synthetic standalone tables — force-emits regardless.
- Once bound, every standalone `Parameter` promotion adds a single-field struct to the same module.

### Parameter dedup (Option 1 — validating, OPNIC-strict)

`Parameter.__new__` looks up the name across `_registry` (parameters inside any registered table) and `_pending_standalone_opnic`:

- **No match** — fresh instance.
- **Match, either side OPNIC** — `ValueError`. OPNIC parameters are single-owner and cannot be re-declared or shared by re-construction.
- **Match, both non-OPNIC** — the existing instance is returned **only after** validating that all requested constructor args are compatible (qua_type, input_type, direction, length, units). On mismatch, a `ValueError` with a per-field diff is raised so the user catches accidental aliasing instead of silently dropping new args.

This replaces the historical "warn-and-return-existing" behaviour, which would silently clobber the new args. (See git history: this fixes the line-274 TODO.)

### Cross-process caveat

The `BaseModule` returned by `to_quarc_module()` is a real in-process Quarc module. If the QUA-side and classical-side code execute in the same Python session (typical for `quarc.run()` driven from a single entry point) they share the same module instance and stream ids match by reference. For multi-process deployments, the classical side must rebuild the wrapper tables — typically by calling `from_quarc_module(my_module)` against the same `BaseModule` subclass / serialized representation. Quarc's own dynamic-module workflow (declaring structs at runtime rather than in static source) only works cleanly when both sides share the assembled module.

In QUA: **declare_variables()**, **load_input_values()** (for OUTGOING tables), **stream_back()** (for INCOMING). From Python: **push_to_opx()** (OUTGOING) or **fetch_from_opx()** (INCOMING).

---

## Quarc hybrid alignment

Authoritative packet layout and stream ids come from **Quarc**. Both pipelines (see ParameterPool section above) end up in the same final state: `table._var` is a `QuaStructHandle`; QUA-side declaration runs through `initialize_in_qua()`; transport flows through `send` / `recv`.

**QUA mapping (same `qm.qua` primitives as `QuaStructHandle`):**

| `ParameterTable` (OPNIC) | Quarc `QuaStructHandle` |
|--------------------------|-------------------------|
| `declare_variables()` (`declare_struct` + `declare_external_stream`) | `initialize_in_qua()` |
| `load_input_values()` → `receive_from_external_stream` | `recv()` |
| `stream_back()` → `send_to_external_stream` | `send()` |

Directions follow the same **`Direction`** enum as elsewhere in this doc (classical-centric: **OUTGOING** = classical → OPX, **INCOMING** = OPX → classical).

For every OPNIC table currently registered in the pool, call **`ParameterPool.iter_opnic_parameter_tables()`**—sorted by table `_id`. For every standalone OPNIC parameter (pending or promoted), call **`ParameterPool.iter_standalone_opnic_parameters()`**.

### Lifecycle walkthrough — Pipeline 2 (parameters-first)

```python
# Phase 0. Empty pool.
ParameterPool.reset()
# pool._registry = {}, pool._pending_standalone_opnic = [], pool._quarc_module = None

# Phase 1. Declare a multi-field OPNIC ParameterTable.
mu    = Parameter("mu", [0.]*4, input_type=OPNIC, direction=OUTGOING)
sigma = Parameter("sigma", [.1]*4, input_type=OPNIC, direction=OUTGOING)
# pool._pending_standalone_opnic = [mu, sigma]
table = ParameterTable([mu, sigma], name="Policy")
# pool._registry = {1: table}; mu and sigma removed from pending (set_index call).
# table._struct_type built (cheap), table._is_emitted=False, table._var=None.

# Phase 2. Declare a standalone OPNIC Parameter.
solo = Parameter("theta", 0., input_type=OPNIC, direction=OUTGOING)
# pool._pending_standalone_opnic = [solo]; solo is still attachable to any future
# ParameterTable.

# Phase 3. Summon the module.
m = ParameterPool.to_quarc_module()
# pool._quarc_module = m (a fresh BaseModule). table is swept onto m via add_struct
# (one stream id consumed). table._is_emitted=True, table._var bound to the handle.
# solo is *not* swept — pending standalone parameters survive to_quarc_module().

# Phase 4. Use solo inside QUA.
with program() as prog:
    solo.declare_variable()
    # Promotion: a synthetic ParameterTable named "theta" is built. solo removed from
    # pending. Synthetic table eagerly emits onto m (one more stream id consumed).
    # solo._main_table = synthetic; solo.opnic_table = synthetic.
    # synthetic_table.declare_variables() runs: m._struct_handles[-1].initialize_in_qua().

# Phase 5. From here on, attempts to attach `solo` to another ParameterTable raise.
ParameterTable([solo], name="other")  # ValueError: standalone-locked.
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
# - extra was pending; now swept onto my_module.add_struct (one stream id consumed),
#   extra._is_emitted=True, extra._var bound.

# After this, any new OPNIC ParameterTable / standalone declare_variable() emits
# eagerly onto my_module — same shared accumulator.
```

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
