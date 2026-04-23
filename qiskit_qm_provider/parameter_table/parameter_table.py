# Copyright 2026 Arthur Strauss
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ParameterTable: mapping of parameters to their to-be-declared QUA variables.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

import copy
import warnings
from typing import (
    Any,
    Optional,
    List,
    Dict,
    Union,
    Tuple,
    Literal,
    Callable,
    Type,
    TYPE_CHECKING,
)
import numpy as np
from qm import QuantumMachine
from qm.api.v2.job_api import JobApi
from qm.jobs.running_qm_job import RunningQmJob
from qm.qua import assign, pause, declare, fixed
from qm.qua.type_hints import ResultStreamSource
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
                        Optional[
                            Union[
                                Literal["INPUT_STREAM", "OPNIC", "IO1", "IO2"],
                                InputType,
                            ]
                        ],
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
        self._packet = None
        self._direction = None
        self._usable_for_opnic_communication = False
        self._incoming_stream_id: Optional[int] = None
        self._outgoing_stream_id: Optional[int] = None
        #: Quarc ``QuaStructHandle`` (or pybind runtime endpoint) exposing ``send`` / ``recv`` / field access;
        #: set by :meth:`ParameterPool.from_quarc_module` on the QUA side and rebound to the OPNIC runtime
        #: endpoint (``runtime.<snake_case_struct>``) before calling :meth:`push_to_opx` / :meth:`fetch_from_opx`
        #: on the classical side. All struct transport flows through this single handle.
        self._var: Optional[Any] = None

        if isinstance(parameters_dict, Dict):
            for index, (parameter_name, parameter) in enumerate(
                parameters_dict.items()
            ):
                input_type = None
                direction = None
                if isinstance(parameter, Tuple):
                    assert len(parameter) <= 4, "Invalid format for parameter value."
                    assert isinstance(
                        parameter[0], (int, float, bool, List, np.ndarray)
                    ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."
                    if len(parameter) >= 2:
                        assert (
                            isinstance(parameter[1], (str, type))
                            or parameter[1] is None
                            or parameter[1] == fixed
                        ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."

                    if len(parameter) >= 3:
                        input_type = (
                            InputType(parameter[2])
                            if isinstance(parameter[2], str)
                            else parameter[2]
                        )
                        if self._input_type is None:
                            self._input_type = input_type
                        elif self._input_type != input_type:
                            raise ValueError(
                                "All parameters in the table must have the same input type."
                            )
                        if input_type == InputType.OPNIC:
                            assert (
                                len(parameter) == 4
                            ), "Direction of the parameter is missing (required for OPNIC input)."
                            direction = (
                                Direction(parameter[3])
                                if isinstance(parameter[3], str)
                                else parameter[3]
                            )
                            if self._direction is None:
                                self._direction = direction
                            elif self._direction != direction:
                                raise ValueError(
                                    "All parameters in the table must have the same direction."
                                )

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
                if (
                    getattr(parameter, "opnic_table", None) is not None
                    and parameter.opnic_table is not self
                ):
                    raise ValueError(
                        f"Parameter {parameter.name!r} is already finalized with standalone "
                        "OPNIC ownership and cannot be attached to a new ParameterTable."
                    )
                self.table[parameter.name] = parameter
                self.table[parameter.name].set_index(self, index)
                if self._input_type is None:
                    self._input_type = parameter.input_type
                elif self._input_type != parameter.input_type:
                    raise ValueError(
                        "All parameters in the table must have the same input type."
                    )
                if self._input_type == InputType.OPNIC:
                    if self._direction is None:
                        self._direction = parameter.direction
                    elif self._direction != parameter.direction:
                        raise ValueError(
                            "All parameters in the table must have the same direction."
                        )

    def _sync_stream_ids_from_handle(self, handle: Any) -> None:
        """Hydrate stream ids from Quarc handle._struct_spec when available."""
        struct_spec = getattr(handle, "_struct_spec", None)
        if struct_spec is None:
            return
        incoming = getattr(struct_spec, "incoming_stream_spec", None)
        outgoing = getattr(struct_spec, "outgoing_stream_spec", None)
        self._incoming_stream_id = getattr(incoming, "id", None)
        self._outgoing_stream_id = getattr(outgoing, "id", None)

    def declare_variables(
        self, pause_program=False, declare_streams=True
    ) -> QuaVariable | List[QuaVariable | QuaArrayVariable]:
        """
        QUA Macro to declare all QUA variables associated with the parameter table.
        Should be called at the beginning of the QUA program.
        Args:
            pause_program: Boolean indicating if the program should pause after declaring the variables.
            declare_streams: Boolean indicating if output streams should be declared for all the parameters.

        """
        if self.input_type == InputType.OPNIC:
            if self._var is None:
                raise RuntimeError(
                    f"ParameterTable {self.name!r} has no Quarc struct handle. Build it via "
                    "ParameterPool.from_quarc_module(module) before declaring QUA variables."
                )
            self._var.initialize_in_qua()
            self._packet = self._var.qua_struct

            for parameter in self.parameters:
                if parameter.is_declared:
                    main_table_name = (
                        parameter.main_table.name
                        if parameter.main_table is not None
                        else self.name
                    )
                    raise ValueError(
                        f"Parameter {parameter.name} already declared. "
                        f"It was declared through table '{main_table_name}'."
                    )
                var = getattr(self._packet, parameter.name)
                parameter._var =  var if parameter.is_array else var[0]
                parameter._is_declared = True
                parameter._main_table = self
                if declare_streams:
                    parameter.declare_stream()
                if parameter.is_array:
                    parameter._ctr = declare(int)

            if self._direction == Direction.INCOMING:
                # OPX -> classical: QUA initializes the outgoing packet with default values.
                for parameter in self.parameters:
                    if parameter.is_array:
                        for i in range(parameter.length):
                            assign(parameter.var[i], parameter.value[i])
                    else:
                        assign(parameter.var, parameter.value)

            if pause_program:
                pause()

            self._usable_for_opnic_communication = True
            return self._packet

        else:
            for parameter in self.parameters:
                if parameter.is_declared:
                    warnings.warn(f"Variable {parameter.name} already declared.")
                    continue
                parameter.declare_variable(declare_stream=declare_streams)
            if pause_program:
                pause()
            if len(self.variables) == 1:
                return self.variables[0]
            else:
                return self.variables

    def declare_streams(self) -> List[ResultStreamSource]:
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
                    f"Stream for parameter {parameter.name} already declared. "
                    "Skipping stream declaration."
                )

        return streams

    def load_input_values(
        self, filter_function: Optional[Callable[[Parameter], bool]] = None
    ):
        """
        QUA Macro to load all the input values of the parameters in the parameter table.
        This macro is expected to work jointly with the use of push_to_opx method on the
        Python side.
        Args: filter_func: Optional function to filter the parameters to be loaded.
        """
        if self.input_type == InputType.OPNIC:
            if not self._usable_for_opnic_communication:
                raise ValueError(
                    "Parameter table not usable for OPNIC communication, as it contains parameters that "
                    "were either undeclared or declared through a different table forming main communication packet."
                )
            if filter_function is not None:
                warnings.warn(
                    "Filter function is not supported for OPNIC parameter tables."
                )
            if self.direction == Direction.INCOMING:
                raise ValueError(
                    "Cannot load input values for outgoing OPNIC parameter tables."
                )
            elif (
                self.direction == Direction.OUTGOING or self.direction == Direction.BOTH
            ):
                self._var.recv()

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
                Dict[
                    Union[str, Parameter],
                    Union[Tuple[int, ...], int, Literal["default"]],
                ],
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
                        raise ValueError(
                            f"Parameter {parameter.name} not found in buffering dictionary."
                        )
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
            if (
                isinstance(parameter_name, str)
                and parameter_name not in self.table.keys()
            ):
                raise KeyError(
                    f"No parameter named {parameter_name} in the parameter table."
                )
            if isinstance(parameter_name, str):
                self.table[parameter_name].assign(parameter_value)
            else:
                if not isinstance(parameter_name, Parameter):
                    raise ValueError(
                        "Invalid parameter name. Please use a string or a ParameterValue object."
                    )
                assert (
                    parameter_name in self.parameters
                ), "Provided ParameterValue not in this ParameterTable."
                parameter_name.assign(parameter_value)

    def print_parameters(self):
        """
        Print the parameters in the parameter table.
        """
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
                raise KeyError(
                    f"No parameter named {parameter} in the parameter table."
                )
            return self.table[parameter].type
        elif isinstance(parameter, int):
            for param in self.parameters:
                if param.get_index(self) == parameter:
                    return param.type
            raise IndexError(
                f"No parameter with index {parameter} in the parameter table."
            )
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
            return (
                parameter_name.get_index(self)
                if parameter_name in self.parameters
                else None
            )
        if parameter_name not in self.table.keys():
            raise KeyError(
                f"No parameter named {parameter_name} in the parameter table."
            )
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
                raise KeyError(
                    f"No parameter named {parameter} in the parameter table."
                )
            return self.table[parameter]
        elif isinstance(parameter, int):
            for param in self.parameters:
                if param.get_index(self) == parameter:
                    return param

            raise IndexError(
                f"No parameter with index {parameter} in the parameter table."
            )
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

    def get_variable(
        self, parameter: Union[str, int, Parameter]
    ) -> QuaVariable | QuaArrayVariable:
        """
        Get the QUA variable corresponding to the specified parameter name.

        Args: parameter: Name or index (within the current table) of the parameter to be returned.
        Returns: QUA variable corresponding to the parameter name.

        """
        if isinstance(parameter, str):
            try:
                return self.table[parameter].var
            except KeyError:
                raise KeyError(
                    f"No parameter named {parameter} in the parameter table."
                )

        if isinstance(parameter, int):
            for param in self.parameters:
                if param.get_index(self) == parameter:
                    return param.var
            raise IndexError(
                f"No parameter with index {parameter} in the parameter table."
            )
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
            raise ValueError(
                "Cannot add parameters to a table that has already been declared."
            )
        if isinstance(parameters, Parameter):
            parameters = [parameters]

        start_idx = len(self.table)
        for i, parameter in enumerate(parameters):
            if not isinstance(parameter, Parameter):
                raise ValueError(
                    "Invalid parameter type. Please use a Parameter object."
                )
            if (
                getattr(parameter, "opnic_table", None) is not None
                and parameter.opnic_table is not self
            ):
                raise ValueError(
                    f"Parameter {parameter.name!r} is already finalized with standalone "
                    "OPNIC ownership and cannot be attached to this table."
                )
            if parameter.name in self.table.keys():
                raise KeyError(
                    f"Parameter {parameter.name} already exists in the parameter table."
                )
            parameter.set_index(self, start_idx + i)
            if parameter.input_type != self.input_type:
                raise ValueError(
                    "All parameters in the table must have the same input type."
                )

            self.table[parameter.name] = parameter

    def remove_parameter(self, parameter_value: Union[str, Parameter]):
        """
        Remove a parameter from the parameter table.
        Args: parameter_value: Name of the parameter to be removed or ParameterValue object to be removed.
        """
        if self.is_declared:
            raise ValueError(
                "Cannot remove parameters from a parameter table that has already been declared."
            )
        if isinstance(parameter_value, str):
            if parameter_value not in self.table.keys():
                raise KeyError(
                    f"No parameter named {parameter_value} in the parameter table."
                )
            del self.table[parameter_value]
        elif isinstance(parameter_value, Parameter):
            if parameter_value not in self.parameters:
                raise KeyError("Provided ParameterValue not in this ParameterTable.")
            del self.table[parameter_value.name]
        else:
            raise ValueError(
                "Invalid parameter name. Please use a string or a ParameterValue object."
            )

    def add_table(
        self, parameter_table: Union[List["ParameterTable"], "ParameterTable"]
    ) -> None:
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
                "Invalid parameter table. Please use a ParameterTable object "
                "or a list of ParameterTable objects."
            )

    def __contains__(self, item: str | Parameter):
        if isinstance(item, str):
            return item in self.table.keys()
        elif isinstance(item, Parameter):
            return item in self.parameters
        else:
            raise ValueError(
                "Invalid parameter name. Please use a string or a Parameter object."
            )

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
            raise ValueError(
                "Not all parameters have been declared. Please declare all parameters first."
            )
        return {
            parameter_name: parameter.var
            for parameter_name, parameter in self.table.items()
        }

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
        if not self.input_type == InputType.OPNIC:
            raise ValueError("No packet declared for non-OPNIC parameter tables.")
        if not self.is_declared:
            raise ValueError("Table not declared. Please declare the table first.")
        return self._packet

    @property
    def opnic_struct(self):
        """
        Get the struct type of the parameter table.
        Relevant for OPNIC parameter tables.
        """
        if not self.input_type == InputType.OPNIC:
            raise ValueError("No struct declared for non-OPNIC parameter tables.")
        return self._packet_type

    @property
    def incoming_stream_id(self) -> int:
        """Incoming stream id for this OPNIC table (Quarc incoming spec id)."""
        if self.input_type != InputType.OPNIC:
            raise ValueError(
                "Incoming stream ID is only defined for OPNIC parameter tables."
            )
        return (
            self._incoming_stream_id
            if self._incoming_stream_id is not None
            else self._id
        )

    @property
    def outgoing_stream_id(self) -> int:
        """Outgoing stream id for this OPNIC table (Quarc outgoing spec id)."""
        if self.input_type != InputType.OPNIC:
            raise ValueError(
                "Outgoing stream ID is only defined for OPNIC parameter tables."
            )
        return (
            self._outgoing_stream_id
            if self._outgoing_stream_id is not None
            else self._id
        )

    @property
    def direction(self) -> Direction | None:
        """
        Get the direction of the parameter table.
        Relevant for OPNIC parameter tables.
        "INCOMING": OPX -> OPNIC
        "OUTGOING": OPNIC -> OPX
        "BOTH": OPNIC <-> OPX
        Returns: Direction of the parameter table. None if the parameter table is not an OPNIC parameter table.

        """
        return self._direction

    def push_to_opx(
        self,
        param_dict: Dict[
            Union[str, Parameter], Union[float, int, bool, List, np.ndarray]
        ],
        job: Optional[RunningQmJob | JobApi] = None,
        qm: Optional[QuantumMachine] = None,
        verbosity: int = 1,
    ):
        """
        Client function: Push the values of the parameters to the OPX (Python side).

        For OPNIC tables, each parameter field is assigned onto the bound Quarc struct handle
        (``self._var``) and :meth:`QuaStructHandle.send`-equivalent is invoked. The handle is
        normally the OPNIC runtime endpoint (``runtime.<snake_case_struct>``) on the classical side.

        Args:
            param_dict: Dictionary of the form ``{parameter_name: parameter_value}``.
            job: ``RunningQmJob`` (only needed for IO/input-stream parameters).
            qm: ``QuantumMachine`` (only needed for IO parameters).
            verbosity: Verbosity level of the pushing process.
        """
        if self.input_type != InputType.OPNIC:
            for parameter, value in param_dict.items():
                self.get_parameter(parameter).push_to_opx(value, job, qm, verbosity)
            return

        if self._var is None:
            raise RuntimeError(
                f"ParameterTable {self.name!r} has no OPNIC endpoint; bind one via "
                "QMEnvironment.from_quarc_module or ParameterPool.from_quarc_module before pushing."
            )
        if self.direction == Direction.INCOMING:
            raise ValueError("Cannot push values to incoming OPNIC parameter tables.")

        for p in self.parameters:
            if p in param_dict:
                value = param_dict[p]
            elif p.name in param_dict:
                value = param_dict[p.name]
            else:
                raise KeyError(
                    f"Parameter '{p.name}' not found in the input dictionary; "
                    "all packet fields must be provided."
                )
            if isinstance(value, np.ndarray):
                value = value.tolist()
            setattr(self._var, p.name, value)

        self._var.send()

        if verbosity > 1:
            print(f"Sent packet {self.name!r} via {type(self._var).__name__}.")

    def stream_back(self, reset: bool = False):
        """
        QUA Macro: Stream the values of the parameters to Python.
            This method is used as a QUA macro to send the values of the parameters to the client/server side.
            It is expected to work jointly with the use of fetch_from_opx method on the client side.

        Args:
            reset: Whether to reset the parameter to a 0 value (in the appropriate QUA type) after sending it to the client/server side.
        """
        if self.input_type != InputType.OPNIC:
            for parameter in self.parameters:
                parameter.stream_back(reset)
        else:
            if not self._usable_for_opnic_communication:
                raise ValueError(
                    "Parameter table not usable for OPNIC communication, as it contains parameters that "
                    "were either undeclared or declared through a different table forming main communication packet."
                )
            if self.direction == Direction.OUTGOING:
                raise ValueError(
                    "Cannot send values to outgoing OPNIC parameter tables."
                )

            self._var.send()

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
        param_dict: Dict[str, Any] = {}
        if self.input_type == InputType.OPNIC:
            if self._var is None:
                raise RuntimeError(
                    f"ParameterTable {self.name!r} has no OPNIC endpoint; bind one via "
                    "QMEnvironment.from_quarc_module or ParameterPool.from_quarc_module before fetching."
                )
            if self.direction == Direction.OUTGOING:
                raise ValueError(
                    "Cannot fetch values from outgoing OPNIC parameter tables."
                )

            self._var.recv(fetching_size, fetching_index)
            
            for p in self.parameters:
                param_dict[p.name] = [getattr(self._var, p.name)[i] for i in range(fetching_size)]

        else:
            for parameter in self.parameters:
                value = parameter.fetch_from_opx(
                    job, fetching_index, fetching_size, verbosity, time_out
                )
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
        input_type: Optional[
            Literal["INPUT_STREAM", "OPNIC", "IO1", "IO2"] | InputType
        ] = None,
        filter_function: Optional[Callable[[QiskitParameter | Var], bool]] = None,
        name: Optional[str] = None,
    ) -> Optional["ParameterTable"]:
        """
        Create a ParameterTable object from a QuantumCircuit object (and stores it in circuit metadata).
        This creates a ParameterTable that jointly encapsulates both symbolic (compile-time) parameters and input real-time variables.
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
        param_vector_set = set()
        for parameter in qc.parameters:
            if isinstance(parameter, QiskitParameter):
                if filter_function is not None and not filter_function(parameter):
                    continue
                if isinstance(parameter, ParameterVectorElement):
                    # Qiskit exports vectors as a collection of single parameters, so we need to create a list of parameters for each element of the vector with a unique name.
                    # Transformation rule: p[i] -> _p_i_ where p is the name of the vector and i is the index of the parameter in the vector.
                    param_vec = parameter.vector
                    if param_vec not in param_vector_set:
                        param_vector_set.add(param_vec)
                        param_list.extend(
                            Parameter(
                                f"_{param_vec.name}_{i}_",
                                qua_type=fixed,
                                input_type=input_type,
                                direction=Direction.OUTGOING,
                            )
                            for i in range(len(param_vec))
                        )
                        continue
                    else:
                        continue

                param_list.append(
                    Parameter(
                        parameter.name,
                        qua_type=fixed,
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
                            qua_type=int,
                            input_type=input_type,
                            direction=Direction.OUTGOING,
                        )
                    )
                elif var.type.kind == types.Bool:
                    param_list.append(
                        Parameter(
                            var.name,
                            qua_type=bool,
                            input_type=input_type,
                            direction=Direction.OUTGOING,
                        )
                    )
                else:  # Float
                    param_list.append(
                        Parameter(
                            var.name,
                            qua_type=fixed,
                            input_type=input_type,
                            direction=Direction.OUTGOING,
                        )
                    )
        if len(param_list) == 0:
            return
        new_table = cls(param_list, name if name is not None else qc.name)
        # if "qua" in qc.metadata:
        #     qc.metadata["qua"][new_table.name] = new_table
        # else:
        #     qc.metadata["qua"] = {new_table.name: new_table}

        return new_table

    @classmethod
    def from_other_tables(
        cls,
        tables: Union[List["ParameterTable"], "ParameterTable"],
        name: Optional[str] = None,
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
        QUA Macro: Reset the QUA variables of the parameter table to 0 (in the appropriate QUA type).
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
        for (
            original_param
        ) in self.table.values():  # self.table.values() gives Parameter instances
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
        # - Rebuilding self._packet_type if OPNIC, and updating opnic_struct/stream_id on copied_params.
        # - Initializing self._qua_external_stream and self._packet to None.

        return new_table
