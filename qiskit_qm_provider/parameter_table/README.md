# Parameter and ParameterTable Usage

This document explains how to use the `Parameter` and `ParameterTable` classes provided in the `rl_qoc.qua.parameter_table` module (or similar path). These classes facilitate the management of dynamic parameters within QUA programs, allowing for easier updates and interactions between Python and the OPX, especially for applications like Reinforcement Learning (RL) or Quantum Optimal Control (QOC), including integration with DGX Quantum via OPNIC.


## `Parameter` Class

The `Parameter` class represents a single dynamic parameter that maps to a QUA variable. It handles type inference, declaration within QUA, and mechanisms for updating its value from Python or reading its value back.

### Initialization

You create a `Parameter` instance by providing a name and an initial value. The QUA type (`fixed`, `int`, `bool`) can be inferred from the value or specified explicitly. You can also specify how the parameter interacts with the outside world using `input_type` and `direction` (for DGX Quantum).


## `ParameterTable` Class

The `ParameterTable` groups multiple `Parameter` objects, simplifying their management, especially when they share the same `input_type` (like `DGX_Q`).

### Initialization

The user can declare a `ParameterTable` instance by passing in a dictionary of parameters intended to be updated dynamically, or a list of pre-defined `Parameter` objects.

**Using a Dictionary:**

The dictionary should be of the form:
`{'parameter_name': (initial_value, qua_type, input_type, direction)}`
or simpler forms where types/inputs are inferred or omitted.


*   `qua_type` can be `bool`, `int`, or `fixed`.
*   It is possible to provide a list or a 1D numpy array as `initial_value` to create a QUA array.
*   The optional `input_type` (`InputType.INPUT_STREAM`, `InputType.IO1`, `InputType.IO2`, `InputType.DGX`) specifies how the parameter interacts externally.
*   The optional `direction` (`Direction.INCOMING`, `Direction.OUTGOING`, `Direction.BOTH`) is required if `input_type` is `DGX_Q`. This indicates the designated direction of communication that should be adopted for the packet associated to the `ParameterTable`. We use the following convention:
    1. `Direction.INCOMING`: OPX -> DGX (OPX sends data obtained from the QUA program to the server)
    2. `Direction.OUTGOING`: DGX -> OPX (The server sends parameters that will influence the execution of the QUA program to the OPX)
    3. `Direction.BOTH`: Both prior directions are supported for the same packet (it can be sent and fetched from both devices).
*   If only a value is provided, the type is inferred.

The declaration of the `ParameterTable` can be done as follows:


**Using a List of Parameters:**

This is useful for ensuring consistent `input_type` and `direction`, especially for DGX Quantum.

**Important:** When using `InputType.DGX_Q`, all parameters within a table *must* share the same `direction`.

Once this declaration is done, the provided dictionary or list is converted to a `ParameterTable` instance. The `ParameterTable` class serves as an interface between the QUA program and the Python environment, facilitating the declaration and manipulation of QUA variables.

## Usage in QUA Programs

The `Parameter` and `ParameterTable` classes provide methods (QUA macros) to interact with the parameters within a `with program():` block.

### Declaring Variables

Variables must be declared before use.

**For `ParameterTable`:**
Use `declare_variables()` to declare all parameters in the table. This method should ideally be called at the beginning of the QUA program scope.
*   `declare_variables(pause_program=False, declare_streams=True)`:
    *   `pause_program`: If `True`, pauses the QUA program immediately after declaration.
    *   `declare_streams`: If `True` (default), declares a standard QUA `stream` for each parameter for saving results.
    *   **Returns:** A tuple of the declared QUA variables, or the declared QUA struct if it's a DGX table.

**For individual `Parameter`:**
Use `declare_variable()`.
*   `declare_variable(pause_program=False, declare_stream=True)`: Arguments and return value are similar to the table version, but for a single parameter.

### Assigning Values within QUA

You can change parameter values directly within the QUA program.

**For `ParameterTable`:**
Use `assign_parameters()` for multiple assignments or standard dictionary/attribute assignment for single parameters.
*   `assign_parameters(values: Dict[Union[str, Parameter], ...])`: Assigns multiple values specified in the dictionary.

**For individual `Parameter`:**
Use `assign_value()`.
*   `assign_value(value, is_qua_array=False, condition=None, value_cond=None)`:
    *   `value`: The new value (literal, QUA variable/expression, list, or QUA array).
    *   `condition`: A QUA boolean expression. Assignment only happens if `True`.
    *   `value_cond`: The value assigned if `condition` is `False`. Must be provided if `condition` is provided.

Note that if the provided value is a QUA Array, it is assumed that it has a length that is equal or higher than the current QUA array variable present in the `Parameter`. The assignment of the array is then done with a QUA `for_` loop that iterates over the length of the initial array length and assigns indices based on that. We hence strongly recommend to always use this array assignment with QUA arrays that strictly have the same length as the `Parameter.var` array variable.

### Loading Input Values from Python/External Sources

Use `load_input_values()` (for Table) or `load_input_value()` (for Parameter) to update parameters from their defined input source (`INPUT_STREAM`, `IO1`, `IO2`, `DGX_Q`).
*   `load_input_values()` (Table): Calls `load_input_value()` for each parameter with an `input_type` (unless filtered). For DGX Q OUTGOING/ BOTH tables, receives the packet.
*   `load_input_value()` (Parameter): Behavior depends on `Parameter.input_type`.

### Streaming back QUA values to Python/External Sources

Use `stream_back()` to make the parameter's current value available externally (via IO, DGX_Q, or standard QUA stream).
*   `stream_back()` (Table/Parameter): Behavior depends on `Parameter.input_type`. For DGX_Q INCOMING tables, sends the packet.
`stream_back` has an optional `reset` argument (defaulting to False) that can reset the associated QUA variable/packet to a 0 value in the relevant type of the QUA variable (e.g. 0 for int, 0. for fixed and False for bool).
If input_type is `INPUT_STREAM`, `IO1`or `IO2`, we just use the stream processing, while `DGX_Q`sends the packet to the external stream.

### Saving to QUA Streams

Explicitly save parameter values to standard QUA streams for later retrieval using result analysis tools.
*   `save_to_stream()` (Table/Parameter): Saves current value(s) to associated stream(s).
*   `stream_processing()` (Table/Parameter): Defines how stream data is handled (e.g., `save`, `save_all`).

### Accessing Parameters and Variables

There are methods to access the underlying `Parameter` objects or their QUA variables. Each `Parameter` within a `ParameterTable` also holds an `index` attribute corresponding to its position in the table (starting from 0). This index is particularly useful within QUA for conditional logic, such as selecting which parameter to update using a `switch_` statement.
*   `get_parameter(parameter: Union[str, int]) -> Parameter`: Returns the `Parameter` object by name or index.
*   `parameter_table[key]` or `parameter_table.attribute`: Accesses the QUA variable directly within QUA (after declaration).
*   `parameter.var`: Accesses the QUA variable of a `Parameter` object within QUA (after declaration).
*   `parameter.index`: Returns the integer index of the `Parameter` within the `ParameterTable`. Useful for `switch_` statements in QUA.

## Interaction from Python (Outside QUA)

Use these methods in your Python script to interact with the running QUA program.

### Pushing Values to OPX (`push_to_opx`)

Send data *from* Python *to* the OPX, corresponding to a `load_input_value()` or `load_input_values()` call in QUA.
*   `push_to_opx(value_or_dict, job, verbosity=1)`:
    *   For `Parameter`: `value` is the value/sequence to send.
    *   For `ParameterTable`: `param_dict` maps names to values. For DGX OUTGOING, *all* parameters must be in the dict.
    *   `job`: The `RunningQmJob` object.

### Fetching Values from OPX (`fetch_from_opx`)

Retrieve data *from* the OPX *to* Python, corresponding to `stream_back()` (for IO/DGX_Q) or stream saving (INPUT_STREAM).
*   `fetch_from_opx(job: Optional[RunningQmJob | JobApi] = None,
        fetching_index: int = 0,
        fetching_size: int = 1,
        verbosity: int = 1,
        time_out=30)`:
    *   For `Parameter`: Fetches value based on `input_type`.
    *   For `ParameterTable`: Fetches values based on `input_type`. 
    * `fetching_size` and `fetching_index` can be used to navigate through the data saved in the stream processing with the `save_all` method to facilitate fetching of a particular slice of data. For iterative workloads, we currently require the user to keep track of the indices and fetching size based on the needs of the algorithm (currently no support for backwards navigation in the stream processing).
    Warning: This behavior was not tested with DGX Quantum.
    For DGX_Q INCOMING, reads one packet.
    *   **Returns:** The fetched value/sequence (Parameter) or a dictionary (Table).


## DGX Quantum Integration (`ParameterPool`)

When using `InputType.DGX_Q`, the `Parameter` and `ParameterTable` classes interact with the `ParameterPool`. This pool manages unique stream IDs required for OPNIC communication and handles the necessary patching and configuration of the underlying `opnic_wrapper`.

### Typical Workflow

1.  **Define Parameters/Tables:** Create your `Parameter` or `ParameterTable` instances with `input_type=InputType.DGX_Q` and the correct `direction`. This automatically registers them with the `ParameterPool`.
2.  **Initialize Streams:** Before running the QUA program, call `ParameterPool.initialize_streams()`. This patches the wrapper code and configures the OPNIC streams.
3.  **Execute QUA Job:** Run the QUA program containing `declare_variables()`, `load_input_values()` (for OUTGOING tables), and `send_to_python()` (for INCOMING tables).
4.  **Python Interaction:** Use `table.push_to_opx()` (OUTGOING) or `table.fetch_from_opx()` (INCOMING) in your Python script.
5.  **Close Streams:** After interaction, call `ParameterPool.close_streams()`.

These classes provide a structured and flexible way to handle dynamic parameters in QUA, streamlining the process of updating variables from Python and integrating external hardware like DGX. Remember to manage the lifecycle of DGX streams using DGXParameterPool when applicable.