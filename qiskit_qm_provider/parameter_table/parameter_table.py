"""
Parameter Table: Class enabling the mapping of parameters to be updated to their corresponding
"to-be-declared" QUA variables.

Author: Arthur Strauss - Quantum Machines
Created: 25/11/2024
"""

from __future__ import annotations

import copy
import warnings
import sys
from itertools import chain
from numbers import Number
from typing import Optional, List, Dict, Union, Tuple, Literal, Callable, Type, TYPE_CHECKING
import numpy as np
from qm import QuantumMachine
from qm.api.v2.job_api import JobApi
from qm.jobs.running_qm_job import RunningQmJob
from qm.qua import assign, pause, declare, fixed
from qm.qua._dsl import _ResultSource
from qm.qua._expressions import QuaArrayVariable
from quam.utils.qua_types import QuaVariable

from .parameter_pool import ParameterPool
from .parameter import Parameter
from .input_type import InputType, Direction

if TYPE_CHECKING:
    from qiskit.circuit.classical.expr import Var
    from qiskit.circuit import QuantumCircuit, Parameter as QiskitParameter


class ParameterTable:
    """
    Class enabling the mapping of parameters to be updated to their corresponding "to-be-declared" QUA variables. The
    type of the QUA variable to be adjusted is automatically inferred from the type of the initial_parameter_value.
    Each parameter in the dictionary should be given a name that the user can then easily access through the table
    with table[parameter_name]. Calling this will return the QUA variable built within the QUA program corresponding
    to the parameter name and its associated Python initial value. Args: parameters_dict: Dictionary of the form {
    "parameter_name": initial_parameter_value }. the QUA program.
    """

    def __init__(
        self,
        parameters_dict: Union[
            Dict[
                str,
                Union[
                    Tuple[
                        Union[float, int, bool, List, np.ndarray],
                        Optional[Union[str, type]],
                        Optional[Union[Literal["INPUT_STREAM", "DGX", "IO1", "IO2"], InputType]],
                        Optional[Union[Literal["INCOMING", "OUTGOING"], Direction]],
                    ],
                    Union[float, int, bool, List, np.ndarray],
                ],
            ],
            List[Parameter],
        ],
        name: Optional[str] = None,
    ):
        """
        Class enabling the mapping of parameters to be updated to their corresponding "to-be-declared" QUA variables.
        The type of the QUA variable to be adjusted can be specified or either be automatically inferred from the
        type of the initial_parameter_value. Each parameter in the dictionary should be given a name that the user
        can then easily access through the table with table[parameter_name]. Calling this will return the QUA
        variable built within the QUA program corresponding to the parameter name and its associated Python initial
        value.

        When initialized with a list of Parameter objects, the input type and direction are for all parameters in the
        list should be the same. The input type and direction are inferred from the first parameter in the list.


        Args:
            parameters_dict: Dictionary should be of the form
            { "parameter_name": (initial_value, qua_type, Literal["input_stream"]) }
            where qua_type is the type of the QUA variable to be declared (int, fixed, bool)
             and the last (optional) field indicates if the variable should be declared as an input_stream instead
             of a standard QUA variable.
            There can also be a list of pre-declared Parameter objects.
            name: Optional name for the parameter table


        """
        self.table: Dict[str, Parameter] = {}
        if name is not None:
            self.name = name
        else:  # Generate a unique name
            self.name = f"ParameterTable_{id(self)}"
        self._input_type = None
        self._id = ParameterPool.get_id(self)
        self._qua_external_stream = None
        self._packet = None
        self._packet_type = None
        self._direction = None
        self._usable_for_dgx_communication = False

        if isinstance(parameters_dict, Dict):
            for index, (parameter_name, parameter) in enumerate(parameters_dict.items()):
                input_type = None
                direction = None
                if isinstance(parameter, Tuple):
                    assert len(parameter) <= 4, "Invalid format for parameter value."
                    assert isinstance(
                        parameter[0], (int, float, bool, List, np.ndarray)
                    ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."
                    if len(parameter) >= 2:
                        assert (
                            isinstance(parameter[1], (str, type)) or parameter[1] is None or parameter[1] == fixed
                        ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."

                    if len(parameter) >= 3:
                        input_type = InputType(parameter[2]) if isinstance(parameter[2], str) else parameter[2]
                        if self._input_type is None:
                            self._input_type = input_type
                        elif self._input_type != input_type:
                            raise ValueError("All parameters in the table must have the same input type.")
                        if input_type == InputType.DGX_Q:
                            assert (
                                len(parameter) == 4
                            ), "Direction of the parameter is missing (required for DGX input)."
                            direction = Direction(parameter[3]) if isinstance(parameter[3], str) else parameter[3]
                            if self._direction is None:
                                self._direction = direction
                            elif self._direction != direction:
                                raise ValueError("All parameters in the table must have the same direction.")

                    self.table[parameter_name] = Parameter(
                        parameter_name,
                        parameter[0],
                        parameter[1],
                        input_type,
                        direction,
                    )
                    self.table[parameter_name].set_index(self, index)

                else:
                    assert isinstance(
                        parameter, (int, float, bool, List, np.ndarray)
                    ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."
                    self.table[parameter_name] = Parameter(parameter_name, parameter)
                    self.table[parameter_name].set_index(self, index)
        elif isinstance(parameters_dict, List):
            for index, parameter in enumerate(parameters_dict):
                assert isinstance(
                    parameter, Parameter
                ), "Invalid format for parameter value. Please use Parameter object."
                self.table[parameter.name] = parameter
                self.table[parameter.name].set_index(self, index)
                if self._input_type is None:
                    self._input_type = parameter.input_type
                elif self._input_type != parameter.input_type:
                    raise ValueError("All parameters in the table must have the same input type.")
                if self._input_type == InputType.DGX_Q:
                    if self._direction is None:
                        self._direction = parameter.direction
                    elif self._direction != parameter.direction:
                        raise ValueError("All parameters in the table must have the same direction.")

        if self.input_type == InputType.DGX_Q:
            from qm.qua import qua_struct, QuaArray

            attributes = {
                parameter.name: QuaArray[parameter.type, parameter.length if parameter.is_array else 1]
                for parameter in self.parameters
            }
            struct_name = f"Packet_{self.name}_{self._id}"
            self._packet_type = qua_struct(type(struct_name, (object,), {"__annotations__": attributes}))

    def declare_variables(self, pause_program=False) -> QuaVariable | List[QuaVariable | QuaArrayVariable]:
        """
        QUA Macro to declare all QUA variables associated with the parameter table.
        Should be called at the beginning of the QUA program.
        Args:
            pause_program: Boolean indicating if the program should pause after declaring the variables.
            declare_streams: Boolean indicating if output streams should be declared for all the parameters.

        """
        if self.input_type == InputType.DGX_Q:
            from qm.qua import (
                declare_struct,
                declare_external_stream,
                QuaStreamDirection,
            )

            qua_direction = (
                QuaStreamDirection.INCOMING if self.direction == Direction.OUTGOING else QuaStreamDirection.OUTGOING
            )
            self._packet = declare_struct(self._packet_type)
            self._qua_external_stream = declare_external_stream(self._packet, self._id, qua_direction)

            for parameter in self.parameters:
                # In ParameterTable.declare_variables(), DGX path:
                if parameter.is_declared:
                    main_table_name = "an unknown table (main_table not set)"
                    if parameter.main_table is not None:  # Check if main_table is actually set
                        main_table_name = parameter.main_table.name
                    raise ValueError(
                        f"Parameter {parameter.name} already declared. "
                        f"It was declared through table '{main_table_name}'."
                    )

                parameter._var = self._packet
                parameter._is_declared = True
                parameter.stream_id = self._id
                parameter.dgx_struct = self._packet_type
                parameter._main_table = self

                if parameter.is_array:
                    parameter._ctr = declare(int)

            if self._direction == Direction.INCOMING:  # OPX -> DGX (Initialize the packet)
                for parameter in self.parameters:
                    if parameter.is_array:
                        for i in range(parameter.length):
                            assign(parameter.var[i], parameter.value[i])
                    else:
                        assign(parameter.var, parameter.value)

            if pause_program:
                pause()

            self._usable_for_dgx_communication = True
            return self._packet

        else:
            for parameter in self.parameters:
                if parameter.is_declared:
                    warnings.warn(f"Variable {parameter.name} already declared.")
                    continue
                parameter.declare_variable()
            if pause_program:
                pause()
            if len(self.variables) == 1:
                return self.variables[0]
            else:
                return self.variables

    def declare_streams(self) -> List[_ResultSource]:
        """
        QUA Macro to declare all the output streams associated with the parameters in the parameter table.
        This macro is expected to be called at the beginning of the QUA program.
        """
        streams = []
        for parameter in self.parameters:
            if parameter.stream is None:
                stream = parameter.declare_stream()
                streams.append(stream)
            else:
                warnings.warn(
                    f"Stream for parameter {parameter.name} already declared. " "Skipping stream declaration."
                )

        return streams

    def load_input_values(self, filter_function: Optional[Callable[[Parameter], bool]] = None):
        """
        QUA Macro to load all the input values of the parameters in the parameter table.
        This macro is expected to work jointly with the use of push_to_opx method on the
        Python side.
        Args: filter_func: Optional function to filter the parameters to be loaded.
        """
        if self.input_type == InputType.DGX_Q:
            from qm.qua import receive_from_external_stream

            if not self._usable_for_dgx_communication:
                raise ValueError(
                    "Parameter table not usable for DGX communication, as it contains parameters that "
                    "were either undeclared or declared through a different table forming main communication packet."
                )
            if filter_function is not None:
                warnings.warn("Filter function is not supported for DGX parameter tables.")
            if self.direction == Direction.INCOMING:
                raise ValueError("Cannot load input values for outgoing DGX parameter tables.")
            elif self.direction == Direction.OUTGOING:
                receive_from_external_stream(self._qua_external_stream, self._packet)

        else:
            for parameter in self.parameters:
                if filter_function is None or filter_function(parameter):
                    parameter.load_input_value()

    def save_to_stream(self):
        """
        Save all the parameters in the parameter table to their associated output streams.
        """
        for parameter in self.parameters:
            if parameter.is_declared and parameter.stream is not None:
                parameter.save_to_stream()

    def stream_processing(
        self,
        mode: Literal["save", "save_all"] = "save_all",
        buffering: Optional[
            Union[
                Dict[Union[str, Parameter], Union[Tuple[int, ...], int, Literal["default"]]],
                Literal["default"],
            ]
        ] = "default",
    ):
        """
        Process all the streams in the parameter table.
        """
        for parameter in self.parameters:
            if parameter.stream is not None:
                if buffering is None:
                    buffer = None
                elif buffering != "default":
                    if parameter.name in buffering:
                        key = parameter.name
                    elif parameter in buffering:
                        key = parameter
                    else:
                        raise ValueError(f"Parameter {parameter.name} not found in buffering dictionary.")
                    buffer = buffering[key]
                else:
                    buffer = "default"
                parameter.stream_processing(mode, buffer)

    def assign_parameters(
        self,
        values: Dict[
            Union[str, Parameter],
            Union[int, float, bool, List, np.ndarray, Parameter, QuaVariable],
        ],
    ):
        """
        Assign values to the parameters of the parameter table within the QUA program.
        Args: values: Dictionary of the form { "parameter_name": parameter_value }. The parameter value can be either
        a Python value or a QuaExpressionType.
        """
        for parameter_name, parameter_value in values.items():
            if isinstance(parameter_name, str) and parameter_name not in self.table.keys():
                raise KeyError(f"No parameter named {parameter_name} in the parameter table.")
            if isinstance(parameter_name, str):
                self.table[parameter_name].assign(parameter_value)
            else:
                if not isinstance(parameter_name, Parameter):
                    raise ValueError("Invalid parameter name. Please use a string or a ParameterValue object.")
                assert parameter_name in self.parameters, "Provided ParameterValue not in this ParameterTable."
                parameter_name.assign(parameter_value)

    def print_parameters(self):
        text = ""
        for parameter_name, parameter in self.table.items():
            text += f"{parameter_name}: {parameter.value}, \n"
        print(text)

    def get_type(self, parameter: Union[str, int, Parameter]) -> Type:
        """
        Get the type of a specific parameter in the parameter table (specified by name or index).

        Args: parameter: Name or index (within current table) of the parameter to get the type of.

        Returns: Type of the parameter in the parameter table.
        """
        if isinstance(parameter, str):
            if parameter not in self.table.keys():
                raise KeyError(f"No parameter named {parameter} in the parameter table.")
            return self.table[parameter].type
        elif isinstance(parameter, int):
            for param in self.parameters:
                if param.get_index(self) == parameter:
                    return param.type
            raise IndexError(f"No parameter with index {parameter} in the parameter table.")
        elif isinstance(parameter, Parameter):
            if parameter not in self.parameters:
                raise KeyError("Provided ParameterValue not in this ParameterTable.")
            return parameter.type
        else:
            raise ValueError("Invalid parameter name. Please use a string or an int.")

    def get_index(self, parameter_name: Union[str, Parameter]) -> int:
        """
        Get the index of a specific parameter in the parameter table.
        Args: parameter_name: (Name of the) parameter to get the index of.
        Returns: Index of the parameter in the parameter table.
        """
        if isinstance(parameter_name, Parameter):
            return parameter_name.get_index(self) if parameter_name in self.parameters else None
        if parameter_name not in self.table.keys():
            raise KeyError(f"No parameter named {parameter_name} in the parameter table.")
        return self.table[parameter_name].get_index(self)

    def get_parameter(self, parameter: Union[str, int, Parameter]) -> Parameter:
        """
        Get the Parameter object of a specific parameter in the parameter table.
        This object contains the QUA variable corresponding to the parameter, its type,
        its index within the current table.

        Args: parameter: Name or index (within current table) of the parameter to be returned.

        Returns: Parameter object corresponding to the specified input.
        """
        if isinstance(parameter, str):
            if parameter not in self.table.keys():
                raise KeyError(f"No parameter named {parameter} in the parameter table.")
            return self.table[parameter]
        elif isinstance(parameter, int):
            for param in self.parameters:
                if param.get_index(self) == parameter:
                    return param

            raise IndexError(f"No parameter with index {parameter} in the parameter table.")
        elif isinstance(parameter, Parameter):
            if parameter not in self.parameters:
                raise KeyError("Provided Parameter not in this ParameterTable.")
            return parameter
        else:
            raise ValueError("Invalid parameter name. Please use a string or an int.")

    def has_parameter(self, parameter: Union[str, int, Parameter]) -> bool:
        """
        Check if a parameter is in the parameter table.
        Args: parameter: Name, index or instance of the parameter to be checked.
        Returns: True if the parameter is in the table, False otherwise.
        """
        if isinstance(parameter, str):
            return parameter in self.table.keys()
        elif isinstance(parameter, int):
            for param in self.parameters:
                if param.get_index(self) == parameter:
                    return True
            return False
        elif isinstance(parameter, Parameter):
            return parameter in self.parameters
        else:
            raise ValueError("Invalid parameter name. Please use a string or an int.")

    def get_variable(self, parameter: Union[str, int, Parameter]) -> QuaVariable | QuaArrayVariable:
        """
        Get the QUA variable corresponding to the specified parameter name.

        Args: parameter: Name or index (within the current table) of the parameter to be returned.
        Returns: QUA variable corresponding to the parameter name.

        """
        if isinstance(parameter, str):
            try:
                return self.table[parameter].var
            except KeyError:
                raise KeyError(f"No parameter named {parameter} in the parameter table.")

        if isinstance(parameter, int):
            for param in self.parameters:
                if param.get_index(self) == parameter:
                    return param.var
            raise IndexError(f"No parameter with index {parameter} in the parameter table.")
        if isinstance(parameter, Parameter):
            if parameter not in self.parameters:
                raise KeyError("Provided ParameterValue not in this ParameterTable.")
            return parameter.var

        raise ValueError("Invalid parameter name. Please use a string or an int.")

    def add_parameters(self, parameters: Union[Parameter, List[Parameter]]):
        """
        Add a (list of) parameter(s) to the parameter table. The index of the parameter is automatically set to the
        next available index in the table.
        Args: parameters: (List of) Parameter(s) object(s) to be added to the current parameter table.
        """
        if self.is_declared:
            raise ValueError("Cannot add parameters to a table that has already been declared.")
        if isinstance(parameters, Parameter):
            parameters = [parameters]

        start_idx = len(self.table)
        for i, parameter in enumerate(parameters):
            if not isinstance(parameter, Parameter):
                raise ValueError("Invalid parameter type. Please use a Parameter object.")
            if parameter.name in self.table.keys():
                raise KeyError(f"Parameter {parameter.name} already exists in the parameter table.")
            parameter.set_index(self, start_idx + i)
            if parameter.input_type != self.input_type:
                raise ValueError("All parameters in the table must have the same input type.")

            self.table[parameter.name] = parameter

    def remove_parameter(self, parameter_value: Union[str, Parameter]):
        """
        Remove a parameter from the parameter table.
        Args: parameter_value: Name of the parameter to be removed or ParameterValue object to be removed.
        """
        if self.is_declared:
            raise ValueError("Cannot remove parameters from a parameter table that has already been declared.")
        if isinstance(parameter_value, str):
            if parameter_value not in self.table.keys():
                raise KeyError(f"No parameter named {parameter_value} in the parameter table.")
            del self.table[parameter_value]
        elif isinstance(parameter_value, Parameter):
            if parameter_value not in self.parameters:
                raise KeyError("Provided ParameterValue not in this ParameterTable.")
            del self.table[parameter_value.name]
        else:
            raise ValueError("Invalid parameter name. Please use a string or a ParameterValue object.")

    def add_table(self, parameter_table: Union[List["ParameterTable"], "ParameterTable"]) -> None:
        """
        Add a parameter table to the current table.
        Args: parameter_table: ParameterTable object to be merged with the current table.
        """
        if isinstance(parameter_table, ParameterTable):
            self.add_table([parameter_table])
        elif isinstance(parameter_table, List):
            for table in parameter_table:
                self.add_parameters(table.parameters)

        else:
            raise ValueError(
                "Invalid parameter table. Please use a ParameterTable object " "or a list of ParameterTable objects."
            )

    def __contains__(self, item: str | Parameter):
        if isinstance(item, str):
            return item in self.table.keys()
        elif isinstance(item, Parameter):
            return item in self.parameters
        else:
            raise ValueError("Invalid parameter name. Please use a string or a Parameter object.")

    def __iter__(self):
        return iter(self.table.values())

    def __setitem__(self, key, value):
        """
        Assign values to the parameters of the parameter table within the QUA program.
        Args: key: Name of the parameter to be assigned. value: Value to be assigned to the parameter.
        """
        if key not in self.table.keys():
            raise KeyError(f"No parameter named {key} in the parameter table.")
        self.table[key].assign(value)

    def __getitem__(self, item: Union[str, int]):
        """
        Returns the QUA variable corresponding to the specified parameter name or parameter index.
        """
        if isinstance(item, str):
            if item not in self.table.keys():
                raise KeyError(f"No parameter named {item} in the parameter table.")
            if self.table[item].is_declared:
                return self.table[item].var
            else:
                raise ValueError(
                    f"No QUA variable found for parameter {item}. Please use "
                    f"ParameterTable.declare_variables() within QUA program first."
                )
        elif isinstance(item, int):
            for parameter in self.table.values():
                if parameter.get_index(self) == item:
                    if parameter.is_declared:
                        return parameter.var
                    else:
                        raise ValueError(
                            f"No QUA variable found for parameter with index {item}. Please use "
                            f"ParameterTable.declare_variables() within QUA program first."
                        )
            raise IndexError(f"No parameter with index {item} in the parameter table.")
        else:
            raise ValueError("Invalid parameter name. Please use a string or an int.")

    def __len__(self):
        return len(self.table)

    def __getattr__(self, item):
        # Get the QUA variable corresponding to the specified parameter name.
        if item in self.table.keys():
            return self.table[item].var
        else:
            raise AttributeError(f"No attribute named {item} in the parameter table.")

    @property
    def variables(self):
        """
        List of the QUA variables corresponding to the parameters in the parameter table.
        """

        return [self[item] for item in self.table.keys()]

    @property
    def variables_dict(self) -> Dict[str, QuaVariable | QuaArrayVariable]:
        """Dictionary of the QUA variables corresponding to the parameters in the parameter table."""
        if not self.is_declared:
            raise ValueError("Not all parameters have been declared. Please declare all parameters first.")
        return {parameter_name: parameter.var for parameter_name, parameter in self.table.items()}

    @property
    def parameters_dict(self) -> Dict[str, Parameter]:
        """Dictionary of the parameters in the parameter table."""
        return self.table

    @property
    def parameters(self) -> List[Parameter]:
        """
        List of the parameter values objects in the parameter table.

        Returns: List of ParameterValue objects in the parameter table.
        """
        return list(self.table.values())

    @property
    def is_declared(self) -> bool:
        """Boolean indicating if all the QUA variables have been declared."""
        return all(parameter.is_declared for parameter in self.parameters)

    @property
    def input_type(self) -> InputType:
        return self._input_type

    @property
    def packet(self):
        if not self.input_type == InputType.DGX_Q:
            raise ValueError("No packet declared for non-DGX parameter tables.")
        if not self.is_declared:
            raise ValueError("Table not declared. Please declare the table first.")
        return self._packet
    
    @property   
    def dgx_struct(self):
        """
        Get the struct type of the parameter table.
        Relevant for DGX Quantum parameter tables.
        """
        if not self.input_type == InputType.DGX_Q:
            raise ValueError("No struct declared for non-DGX parameter tables.")
        return self._packet_type

    @property
    def stream_id(self) -> int:
        """
        Get the stream ID of the parameter table.
        Relevant for DGX parameter tables.
        """
        return self._id

    @property
    def direction(self) -> Direction:
        """
        Get the direction of the parameter table.
        Relevant for DGX parameter tables.
        "INCOMING": OPX -> DGX
        "OUTGOING": DGX -> OPX
        Returns:

        """
        if self.input_type != InputType.DGX_Q:
            raise ValueError("Direction is only relevant for DGX parameter tables.")
        return self._direction

    def push_to_opx(
        self,
        param_dict: Dict[Union[str, Parameter], Union[float, int, bool, List, np.ndarray]],
        job: Optional[RunningQmJob | JobApi] = None,
        qm: Optional[QuantumMachine] = None,
        verbosity: int = 1,
    ):
        """
        Client function: Push the values of the parameters to the OPX (Python side).
        Args:
            param_dict: Dictionary of the form {parameter_name: parameter_value}.
            The parameter value can be either a Python value or a QuaExpressionType.
            job: RunningQmJob object to push the values to.
            qm: QuantumMachine object to push the values to (IO variables). Relevant only for OPX+ with IO variables.
            verbosity: Verbosity level of the pushing process.
        """
        if self.input_type != InputType.DGX_Q:
            for parameter, value in param_dict.items():
                self.get_parameter(parameter).push_to_opx(value, job, qm, verbosity)

        else:
            if not self._usable_for_dgx_communication:
                raise ValueError(
                    "Parameter table not usable for DGX communication, as it contains parameters that "
                    "were either undeclared or declared through a different table forming main communication packet."
                )
            if self.direction == Direction.INCOMING:
                raise ValueError("Cannot push values to incoming DGX parameter tables.")
            # Check if all parameters are in the dictionary
            values_for_packet = {}
            for p_obj in self.parameters:
                value = None
                if p_obj in param_dict:
                    value = param_dict[p_obj]
                elif p_obj.name in param_dict:
                    value = param_dict[p_obj.name]
                else:
                    raise KeyError(
                        f"Parameter '{p_obj.name}' not found in the input dictionary;"
                        f"all packet fields must be provided."
                    )
                processed_value = value.tolist() if isinstance(value, np.ndarray) else value
                processed_value = (
                    [processed_value] if not p_obj.is_array and isinstance(processed_value, Number) else processed_value
                )
                values_for_packet[p_obj.name] = processed_value

            if ParameterPool.configured() and ParameterPool.patched():
                from opnic_wrapper import OutgoingPacket, send_packet

                flattened_values = list(chain(*values_for_packet.values()))
                packet = OutgoingPacket(*flattened_values)
                send_packet(self.stream_id, packet)
            else:
                raise ValueError("OPNIC wrapper not configured or patched.")

            if verbosity > 1:
                print(f"Sent packet: {values_for_packet}")

    def stream_back(self, reset: bool = False):
        """
        QUA Macro: Stream the values of the parameters to Python.
            This method is used as a QUA macro to send the values of the parameters to the client/server side.
            It is expected to work jointly with the use of fetch_from_opx method on the client side.

        Args:
            reset: Whether to reset the parameter to a 0 value (in the appropriate QUA type) after sending it to the client/server side.
        """
        if self.input_type != InputType.DGX_Q:
            for parameter in self.parameters:
                parameter.stream_back(reset)
        else:
            if not self._usable_for_dgx_communication:
                raise ValueError(
                    "Parameter table not usable for DGX communication, as it contains parameters that "
                    "were either undeclared or declared through a different table forming main communication packet."
                )
            if self.direction == Direction.OUTGOING:
                raise ValueError("Cannot send values to outgoing DGX parameter tables.")
            from qm.qua import send_to_external_stream

            send_to_external_stream(self._qua_external_stream, self._packet)
            
            for parameter in self.parameters:
                if parameter.stream is not None:
                    parameter.save_to_stream()
                if reset:
                    parameter.reset_var()
            

    def fetch_from_opx(
        self,
        job: Optional[RunningQmJob | JobApi] = None,
        fetching_index: int = 0,
        fetching_size: int = 1,
        verbosity: int = 1,
        time_out: int = 30,
    ):
        """
        Client function: Fetch the values of the parameters from the OPX (Client/server side).
        The values are returned in a dictionary of the form {parameter_name: parameter_value}.

        Args: job: RunningQmJob object to fetch the values from (input stream).
                qm: QuantumMachine object to fetch the values from (IO variables).
                verbosity: Verbosity level of the fetching process.

        Returns: Dictionary of the form {parameter_name: parameter_value}.
        """
        param_dict = {}
        if self.input_type == InputType.DGX_Q:
            if not self._usable_for_dgx_communication:
                raise ValueError(
                    "Parameter table not usable for DGX communication, as it contains parameters that "
                    "were either undeclared or declared through a different table forming main communication packet."
                )
            if self.direction == Direction.OUTGOING:
                raise ValueError("Cannot fetch values from outgoing DGX parameter tables.")
            elif not ParameterPool.configured() or not ParameterPool.patched():
                raise ValueError("OPNIC wrapper not configured or patched. ")

            from opnic_wrapper import read_packet, wait_for_packets

            wait_for_packets(self.stream_id, fetching_index + fetching_size)
            packets = []
            for i in range(fetching_size):
                packet = read_packet(self.stream_id, fetching_index + i)  # TODO: check if this is correct
                packets.append(packet)
            for parameter in self.parameters:
                param_dict[parameter.name] = np.array([getattr(packet, parameter.name) for packet in packets])

        else:
            for parameter in self.parameters:
                value = parameter.fetch_from_opx(job, fetching_index, fetching_size, verbosity, time_out)
                param_dict[parameter.name] = value
        return param_dict

    def __repr__(self):
        text = "ParameterTable("
        text += f"{self.name}: "
        text += "{"
        for i, parameter in enumerate(self.table.values()):
            text += parameter.__repr__()
            text += ", " if i != len(self.table) - 1 else ""

        text += ")"
        text += "}"
        return text

    @classmethod
    def from_qiskit(
        cls,
        qc: QuantumCircuit,
        input_type: Optional[Literal["INPUT_STREAM", "DGX", "IO1", "IO2"] | InputType] = None,
        filter_function: Optional[Callable[[QiskitParameter | Var], bool]] = None,
        name: Optional[str] = None,
    ) -> Optional["ParameterTable"]:
        """
        Create a ParameterTable object from a QuantumCircuit object (and stores it in circuit metadata).
        Returns None if no parameters are found.
        Args:
            qc: QuantumCircuit object to be converted to a ParameterTable object.
            input_type: Input type of the parameters in the table.
            filter_function: Optional function to filter the parameters to be included in the table.
            name: Optional name for the parameter table.
        """
        from qiskit.circuit import QuantumCircuit, Parameter as QiskitParameter
        from qiskit.circuit.parametervector import ParameterVectorElement
        from qiskit.circuit.classical import types

        param_list = []
        for parameter in qc.parameters:
            if isinstance(parameter, ParameterVectorElement):
                raise ValueError(
                    "ParameterVectors are not yet supported "
                    "(Reason: Qiskit exporter to OpenQASM3 does not "
                    "support array of parameters specification."
                    " Please use a list of individual parameters instead."
                )
            if isinstance(parameter, QiskitParameter):
                if filter_function is not None and not filter_function(parameter):
                    continue
                param_list.append(
                    Parameter(
                        parameter.name,
                        0.0,
                        input_type=input_type,
                        direction=Direction.OUTGOING,
                    )
                )
        if isinstance(qc, QuantumCircuit):
            for var in qc.iter_input_vars():
                if filter_function is not None and not filter_function(var):
                    continue
                if var.type.kind == types.Uint:
                    param_list.append(
                        Parameter(
                            var.name,
                            0,
                            input_type=input_type,
                            direction=Direction.OUTGOING,
                        )
                    )
                elif var.type.kind == types.Bool:
                    param_list.append(
                        Parameter(
                            var.name,
                            False,
                            input_type=input_type,
                            direction=Direction.OUTGOING,
                        )
                    )
                else:  # Float
                    param_list.append(
                        Parameter(
                            var.name,
                            0.0,
                            input_type=input_type,
                            direction=Direction.OUTGOING,
                        )
                    )
        if len(param_list) == 0:
            return
        new_table = cls(param_list, name if name is not None else qc.name)
        if "qua" in qc.metadata:
            qc.metadata["qua"][new_table.name] = new_table
        else:
            qc.metadata["qua"] = {new_table.name: new_table}

        return new_table

    @classmethod
    def from_other_tables(
        cls, tables: Union[List["ParameterTable"], "ParameterTable"], name: Optional[str] = None
    ) -> "ParameterTable":
        """
        Create a ParameterTable object from a list of other ParameterTable objects.
        Args: tables: List of ParameterTable objects to be merged into a new table.
        """
        if isinstance(tables, ParameterTable):
            tables = [tables]
        if not tables:
            raise ValueError("No parameter tables provided.")
        parameters = []
        ref_input_type = tables[0].input_type
        for table in tables:
            if table.input_type != ref_input_type:
                raise ValueError("All parameter tables must have the same input type.")
            parameters.extend(table.parameters)

        new_table = cls(
            list(set(parameters)),
            name if name is not None else "_".join([table.name for table in tables]),
        )

        return new_table

    def reset(self):
        """
        Client function: Reset the parameter table to its initial state.
        """

        for parameter in self.parameters:
            parameter.reset()

    def reset_vars(self):
        """
        QUA Macro:Reset the QUA variables of the parameter table to 0 (in the appropriate QUA type).
        """
        for parameter in self.parameters:
            parameter.reset_var()

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        # Prevent infinite recursion
        if id(self) in memo:
            return memo[id(self)]

        # 1. Create deep copies of all Parameter objects
        copied_parameters_list = []
        for original_param in self.table.values():  # self.table.values() gives Parameter instances
            copied_param = copy.deepcopy(original_param, memo)
            copied_parameters_list.append(copied_param)

        # 2. Create a new ParameterTable instance using its __init__ method.
        #    This leverages the existing initialization logic.
        cls = self.__class__

        # Create the new instance object first
        new_table = cls.__new__(cls)
        # Add to memo *before* calling __init__ to handle potential recursion during __init__
        memo[id(self)] = new_table

        # Prepare arguments for __init__
        new_table_name = self.name + "_copy"  # Or a more sophisticated naming scheme

        # Call __init__ on the newly created instance.
        # Your __init__ already handles List[Parameter] input.
        new_table.__init__(parameters_dict=copied_parameters_list, name=new_table_name)

        # The __init__ method should have handled:
        # - Assigning a new self._id from ParameterPool.
        # - Populating self.table with copied_parameters_list.
        # - Calling param.set_index(new_table, ...) for each copied_param.
        # - Setting self._input_type and self._direction.
        # - Rebuilding self._packet_type if DGX, and updating dgx_struct/stream_id on copied_params.
        # - Initializing self._qua_external_stream and self._packet to None.

        return new_table
