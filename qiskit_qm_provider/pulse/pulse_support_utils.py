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

"""Pulse support: parameterized channel handling, schedule-to-QUA macros, validation.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit.circuit.parametertable import ParameterView
from qiskit.circuit.parametervector import ParameterVectorElement
from quam.components import Channel as QuAMChannel
from .sympy_to_qua import sympy_to_qua
from qiskit.circuit.parameterexpression import ParameterExpression
from ..parameter_table import ParameterTable, Parameter as QuaParameter, InputType
from typing import Dict, Optional, Callable, TYPE_CHECKING
from inspect import Signature, Parameter as sigParam
from qm.qua import switch_, case_

try:
    from qiskit.pulse import (
        Play,
        Instruction,
        Schedule,
        ScheduleBlock,
        Acquire,
    )
    from qiskit.pulse.transforms import block_to_schedule
    from qiskit.pulse.library.pulse import Pulse as QiskitPulse

except ImportError:
    raise ImportError(
        "Failed to import Qiskit Pulse. Please ensure your Qiskit version is below 2.0.0."
    )

from .pulse_to_qua import *

if TYPE_CHECKING:
    from ..backend.qm_backend import QMBackend

# TODO: Add duration to the list of real-time parameters (need ScheduleBlock to QUA compiler)
_real_time_parameters = {
    "amp",
    "angle",
    "frequency",
    "phase",
    "duration",
}  # Parameters that can be used in real-time


def get_real_time_pulse_parameters(pulse: QiskitPulse):
    """
    Get the real-time parameters of a Qiskit Pulse, that is parameters of the pulse that are
    not known at compile time and need to be calculated at runtime (i.e., ParameterExpressions)
    """
    real_time_params = {}
    for param in _real_time_parameters:
        if hasattr(pulse, param) and isinstance(
            getattr(pulse, param), ParameterExpression
        ):
            real_time_params[param] = getattr(pulse, param)
    return real_time_params


def _handle_parameterized_instruction(
    instruction: Instruction,
    param_table: ParameterTable,
    qua_pulse_macro: QuaPulseMacro,
):
    """
    Handle the conversion of a parameterized instruction to QUA by creating a dictionary of parameter values
    and assigning them to the corresponding QUA variables
    """
    if not type(instruction) in qiskit_to_qua_instructions:
        raise ValueError(f"Instruction {instruction} not supported on QM backend")
    value_dict = {}
    involved_parameters = {}  # Store involved Parameters
    validate_parameters(instruction.parameters, param_table)
    for param in instruction.parameters:
        involved_parameters[param.name] = param_table.get_parameter(param.name)

    for attribute in qua_pulse_macro.params:
        attribute_value = getattr(instruction, attribute)
        if isinstance(attribute_value, ParameterExpression):
            value_dict[attribute] = sympy_to_qua(
                getattr(instruction, attribute).sympify(), involved_parameters
            )
        elif attribute == "pulse":
            pulse = getattr(instruction, attribute)
            for pulse_param_name, pulse_param in get_real_time_pulse_parameters(
                pulse
            ).items():
                value_dict[pulse_param_name] = sympy_to_qua(
                    pulse_param.sympify(),
                    involved_parameters,
                )
            break
        # else:  # TODO: Check if this is necessary
        #     value_dict[attribute] = getattr(
        #         instruction, attribute
        #     )  # Assign the value of the attribute

    return value_dict


def validate_pulse(pulse: QiskitPulse, channel: QuAMChannel) -> QiskitPulse:
    """
    Validate the pulse on the QuAM channel
    """
    if not pulse.name in channel.operations:
        raise ValueError(
            f"Pulse {pulse.name} is not in the operations of the QuAM channel"
        )

    return pulse


def validate_instruction(
    instruction: Instruction, quam_channel: QuAMChannel
) -> QuaPulseMacro:
    """
    Validate the instruction before converting it to QUA and return the corresponding QUA macro
    """
    kwargs: Dict[str, QuAMChannel | QiskitPulse] = {"channel": quam_channel}
    if isinstance(instruction, Play):
        pulse = instruction.pulse
        kwargs["pulse"] = pulse
    if type(instruction) in qiskit_to_qua_instructions:
        return qiskit_to_qua_instructions[type(instruction)](**kwargs)
    elif isinstance(instruction, Acquire):
        raise NotImplementedError("Acquire instructions are not yet supported.")
    else:
        raise ValueError(f"Instruction {instruction} not supported on QM backend")


def validate_parameters(
    params: ParameterView, param_table: ParameterTable, param_mapping=None
) -> ParameterTable:
    """
    Validate the parameters of the instruction by checking them against the parameter table
    and a possible parameter mapping

    Args:
        params: List of parameters to validate (names or Qiskit Parameter objects)
        param_table: Parameter table to check the parameters against
        param_mapping: Mapping of parameters to QUA variables
    """
    if not isinstance(param_table, ParameterTable):
        raise ValueError("Parameter table must be provided")
    for param in params:
        if isinstance(param, ParameterVectorElement):
            param_name = param.vector.name
        elif isinstance(param, Parameter):
            param_name = param.name
        else:
            param_name = param
        if param_name not in param_table:
            raise ValueError(f"Parameter {param_name} is not in the parameter table")
        if param_mapping is not None and param_name not in param_mapping:
            raise ValueError(
                f"Parameter {param_name} is not in the provided parameters mapping"
            )
    return param_table


def _instruction_to_qua(
    instruction: Instruction,
    quam_channel: QuAMChannel,
    param_table: Optional[ParameterTable] = None,
):
    """
    Convert a Qiskit Pulse instruction to a QUA instruction

    Args:
        instruction: Qiskit Pulse instruction to convert
        quam_channel: QuAM channel on which to apply the instruction
        param_table:  Parameter table to use for the conversion (contains the reference QUA variables for parameters)
    """

    action = validate_instruction(instruction, quam_channel)

    if instruction.is_parameterized():
        assert param_table is not None, "Parameter table must be provided"
        values = _handle_parameterized_instruction(instruction, param_table, action)

    else:
        values = {}
        if not isinstance(instruction, Play):
            for attribute in action.params:
                values[attribute] = getattr(instruction, attribute)

    action.macro(**values)


def validate_schedule(schedule: Schedule | ScheduleBlock) -> Schedule:
    if isinstance(schedule, ScheduleBlock):
        if not schedule.is_schedulable():
            raise NotImplementedError(
                "ScheduleBlock with parameterized durations are not yet supported"
            )

        schedule = block_to_schedule(schedule)
    if not isinstance(schedule, Schedule):
        raise ValueError("Only Qiskit Pulse Schedule objects are supported")

    return schedule


def handle_parameterized_channel(
    schedule: Schedule, param_table: ParameterTable
) -> ParameterTable:
    """
    Modify type of parameters (-> int) in the Table that refer to channel parameters (they refer to integers)
    """
    for channel in list(filter(lambda ch: ch.is_parameterized(), schedule.channels)):
        ch_params = list(channel.parameters)
        if len(ch_params) > 1:
            raise NotImplementedError(
                "Only single parameterized channels are supported"
            )
        ch_param = ch_params[0]
        if ch_param.name in param_table:
            param_table.get_parameter(ch_param.name).type = int
            param_table.get_parameter(ch_param.name).value = 0
        else:
            ch_parameter_value = QuaParameter(ch_param.name, 0, int)
            param_table.table[ch_param.name] = ch_parameter_value
    return param_table


def schedule_to_qua_macro(
    backend: QMBackend,
    sched: Schedule,
    param_table: Optional[ParameterTable] = None,
    input_type: Optional[InputType] = None,
) -> Callable:
    sig = Signature()
    if sched.is_parameterized():
        if param_table is None:
            param_table = ParameterTable.from_qiskit(
                sched, name=sched.name + "_param_table", input_type=input_type
            )
            param_table = handle_parameterized_channel(sched, param_table)
        else:
            param_table = validate_parameters(sched.parameters, param_table)

        involved_parameters = [value.name for value in sched.parameters]
        params = [
            sigParam(param, sigParam.POSITIONAL_OR_KEYWORD)
            for param in involved_parameters
        ]
        sig = Signature(params)

    def qua_macro(*args, **kwargs):  # Define the QUA macro with parameters
        # Relate passed positional arguments to parameters in ParameterTable
        bound_params = sig.bind(*args, **kwargs)
        bound_params.apply_defaults()
        if param_table is not None:
            for param_name, value in bound_params.arguments.items():
                if not param_table.get_parameter(param_name).is_declared:
                    param_table.get_parameter(param_name).declare_variable()
                param_table.get_parameter(param_name).assign(value)

        time_tracker = {channel: 0 for channel in sched.channels}

        for time, instruction in sched.instructions:
            if len(instruction.channels) > 1:
                raise NotImplementedError(
                    "Only single channel instructions are supported"
                )
            qiskit_channel = instruction.channels[0]

            if (
                qiskit_channel.is_parameterized()
            ):  # Basic support for parameterized channels
                # Filter dictionary of pulses based on provided ChannelType
                channel_dict = {
                    channel.index: quam_channel
                    for channel, quam_channel in backend.channel_mapping.items()
                    if isinstance(channel, type(qiskit_channel))
                }
                ch_parameter_name = list(qiskit_channel.parameters)[0].name
                if not param_table.get_parameter(ch_parameter_name).type == int:
                    raise ValueError(
                        f"Parameter {ch_parameter_name} must be of type int for switch case"
                    )

                # QUA variable corresponding to the channel parameter
                with switch_(param_table[ch_parameter_name]):
                    for i, quam_channel in channel_dict.items():
                        with case_(i):
                            qiskit_channel = backend.get_pulse_channel(quam_channel)
                            if time_tracker[qiskit_channel] < time:
                                quam_channel.wait((time - time_tracker[qiskit_channel]))
                                time_tracker[qiskit_channel] = time
                            _instruction_to_qua(
                                instruction,
                                quam_channel,
                                param_table,
                            )
                            time_tracker[qiskit_channel] += instruction.duration
            else:
                quam_channel = backend.get_quam_channel(qiskit_channel)
                if time_tracker[qiskit_channel] < time:
                    quam_channel.wait((time - time_tracker[qiskit_channel]))
                    time_tracker[qiskit_channel] = time
                _instruction_to_qua(instruction, quam_channel, param_table)
                time_tracker[qiskit_channel] += instruction.duration

    qua_macro.__name__ = sched.name if sched.name else "macro" + str(id(sched))
    qua_macro.__signature__ = sig
    return qua_macro
