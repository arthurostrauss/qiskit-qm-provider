from __future__ import annotations

from typing import Iterable, List, Dict, Optional, Callable, Union, Tuple, Any, TYPE_CHECKING
from inspect import Signature, Parameter as sigParam

from qiskit.circuit import (
    QuantumCircuit,
    Parameter as QiskitParameter,
    Instruction,
)
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.primitives import BitArray, SamplerPubResult, DataBin
from qiskit.providers import BackendV2 as Backend, QubitProperties, Options
from qiskit.pulse import (
    ScheduleBlock,
    Schedule,
    DriveChannel,
    MeasureChannel,
    AcquireChannel,
    Play,
)
from qiskit.pulse.channels import Channel as QiskitChannel, ControlChannel
from qiskit.pulse.library import SymbolicPulse, Waveform, Pulse as QiskitPulse
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

from qiskit.transpiler import Target, InstructionProperties
from qiskit.qasm3 import Exporter
from qm.jobs.running_qm_job import RunningQmJob

# QUA and Quam imports
from qm.qua import *
from qm import QuantumMachinesManager, Program, DictQuaConfig, QuantumMachine
from qm.qua._dsl import _ResultSource
from qualang_tools.addons.variables import assign_variables_to_element
from quam.components import Channel as QuAMChannel, QubitPair
from quam_builder.architecture.superconducting.qpu.base_quam import BaseQuam as Quam
from quam.utils.qua_types import QuaScalar

# OpenQASM3 to QUA compiler
from oqc import (
    Compiler,
    HardwareConfig,
    OperationIdentifier,
    QubitsMapping,
    CompilationResult,
)

# Helper modules
from .parameter_table import ParameterTable, InputType, Parameter
from .pulse_support_utils import (
    _instruction_to_qua,
    validate_parameters,
    validate_schedule,
    handle_parameterized_channel,
)
from .quam_qiskit_pulse import QuAMQiskitPulse, FluxChannel
from .backend_utils import (
    validate_machine,
    look_for_standard_op,
    get_extended_gate_name_mapping,
    has_reset_at_boundary,
    control_flow_name_mapping,
    _QASM3_DUMP_LOOSE_BIT_PREFIX,
    oq3_keyword_instructions,
    validate_circuits,
)
from .qm_instruction_properties import QMInstructionProperties

if TYPE_CHECKING:
    from .qm_job import QMJob
__all__ = [
    "QMBackend",
    "FluxTunableTransmonBackend",
]
RunInput = Union[QuantumCircuit, Schedule, ScheduleBlock]


class QMBackend(Backend):
    def __init__(
        self,
        machine: Quam,
        channel_mapping: Optional[Dict[QiskitChannel, QuAMChannel]] = None,
        init_macro: Optional[Callable] = None,
    ):
        """
        Initialize the QM backend
        Args:
            machine: The Quam instance
            channel_mapping: Optional mapping of Qiskit Pulse Channels to QuAM Channels.
                             This mapping enables the conversion of Qiskit schedules into parametric QUA macros.
            init_macro: Optional macro to be called at the beginning of the QUA program

        """

        Backend.__init__(self, name="QM backend")

        self._custom_instructions = {}
        self.machine = validate_machine(machine)
        self._qmm: Optional[QuantumMachinesManager] = None
        self.channel_mapping: Dict[QiskitChannel, QuAMChannel] = channel_mapping
        self.reverse_channel_mapping: Dict[QuAMChannel, QiskitChannel] = (
            {v: k for k, v in channel_mapping.items()} if channel_mapping is not None else {}
        )
        self._qubit_dict = {qubit.name: i for i, qubit in enumerate(machine.active_qubits)}
        self._target, self._ref_operation_mapping_QUA = self._populate_target(machine)
        self._operation_mapping_QUA = self._ref_operation_mapping_QUA.copy()
        self._oq3_custom_gates = []
        self._init_macro = init_macro

    @property
    def target(self):
        return self._target

    @property
    def custom_instructions(self):
        """
        Get the custom instructions for the backend (those that are part of the target but not in the
        standard Qiskit gate set, inferred from the available macros)
        """
        return self._custom_instructions

    @property
    def qubit_dict(self):
        """
        Get the qubit dictionary for the backend
        """
        return self._qubit_dict

    @property
    def qubit_mapping(self) -> QubitsMapping:
        """
        Build the qubit to quantum elements mapping for the backend.
        Should be of the form {qubit_index: (quantum_element1, quantum_element2, ...)}
        """
        return {
            i: tuple(channel for channel in qubit.channels)
            for i, qubit in enumerate(self.machine.active_qubits)
        }

    @property
    def qubit_index_dict(self):
        """
        Returns a dictionary mapping qubit indices (Qiskit numbering) to corresponding Qubit objects (based on
         the active_qubits attribute of QuAM instance)
        """
        return {i: qubit for i, qubit in enumerate(self.machine.active_qubits)}

    @property
    def qmm(self):
        """
        Returns the QuantumMachinesManager instance. This is a property that reopens a QuantumMachinesManager each time
        it is called for the underlying configuration might have changed between two calls
        """
        if self._qmm is None:
            self._qmm = self.machine.connect()
        return self._qmm

    @property
    def qm(self) -> QuantumMachine:
        """
        Returns the QuantumMachine instance. This is a property that reopens a QuantumMachine each time
        it is called for the underlying configuration might have changed between two calls
        """
        return self.qmm.open_qm(self.qm_config)

    @property
    def qm_config(self) -> DictQuaConfig:
        """
        Returns the QUA configuration for the backend
        """
        return self.machine.generate_config()

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls) -> Options:
        """
        Returns the default options for the backend. The options are:
        - shots: The number of shots to run the circuit for (default is 1024)
        - compiler_options: The options for the QOP compiler (if any)
        - simulate: The simulation configuration to use (if any) on the QOP
        - memory: Whether to save each shot in memory (default is False)
        :return:
        """
        return Options(
            shots=1024,
            compiler_options=None,
            simulate=None,
            memory=False,
            skip_reset=False,
        )

    def _populate_target(self, machine: Quam) -> Tuple[Target, Dict[OperationIdentifier, Callable]]:
        """
        Populate the target instructions with the QOP configuration
        """
        gate_map = get_extended_gate_name_mapping()
        target = Target(
            "Transmon based QuAM",
            dt=1e-9,
            granularity=4,
            num_qubits=len(machine.active_qubits),
            min_length=16,
            qubit_properties=[
                QubitProperties(t1=qubit.T1, t2=qubit.T2ramsey, frequency=qubit.f_01)
                for qubit in machine.active_qubits
            ],
        )

        operations_dict = {}
        operations_qua_dict = {}
        name_to_op_dict = {}

        # Add single qubit instructions
        for q, qubit in enumerate(machine.active_qubits):
            for op, func in qubit.macros.items():
                op_ = look_for_standard_op(op)

                if op_ in gate_map:
                    gate_op = gate_map[op_]
                    num_params = len(gate_op.params)

                    operations_dict.setdefault(op_, {})[(q,)] = None
                    operations_qua_dict[OperationIdentifier(op_, num_params, (q,))] = func.apply
                    name_to_op_dict[op_] = gate_op
                else:
                    # Create custom gate
                    signature = Signature.from_callable(func.apply)
                    params = signature.parameters.values()
                    positional_params = [
                        param
                        for param in params
                        if param.kind in (sigParam.POSITIONAL_OR_KEYWORD, sigParam.POSITIONAL_ONLY)
                    ]

                    params = [QiskitParameter(param.name) for param in positional_params]
                    return_type = signature.return_annotation
                    if return_type is not None and return_type is not Signature.empty:
                        raise ValueError(
                            f"Return type {return_type} not yet supported for custom gate {op_}"
                        )
                    gate_op = Instruction(op_, 1, 0, params)
                    operations_dict.setdefault(op_, {})[(q,)] = None
                    operations_qua_dict[OperationIdentifier(op_, len(params), (q,))] = func.apply
                    name_to_op_dict[op_] = gate_op
                    self._custom_instructions[op_] = gate_op

        for qubit_pair in machine.active_qubit_pairs:
            q_ctrl = self.qubit_dict[qubit_pair.qubit_control.name]
            q_tgt = self.qubit_dict[qubit_pair.qubit_target.name]
            for op, func in qubit_pair.macros.items():
                op_ = look_for_standard_op(op)
                if op_ in gate_map:
                    gate_op = gate_map[op_]
                    num_params = len(gate_op.params)
                    operations_dict.setdefault(op_, {})[(q_ctrl, q_tgt)] = None
                    operations_qua_dict[OperationIdentifier(op_, num_params, (q_ctrl, q_tgt))] = (
                        func.apply
                    )
                    name_to_op_dict[op_] = gate_op
                else:
                    # Create custom gate
                    signature = Signature.from_callable(func.apply)
                    params = signature.parameters.values()
                    positional_params = [
                        param
                        for param in params
                        if param.kind in (sigParam.POSITIONAL_OR_KEYWORD, sigParam.POSITIONAL_ONLY)
                    ]

                    params = [QiskitParameter(param.name) for param in positional_params]
                    return_type = signature.return_annotation
                    if return_type is not None and return_type is not Signature.empty:
                        raise ValueError(
                            f"Return type {return_type} not yet supported for custom gate {op_}"
                        )
                    gate_op = Instruction(op_, 2, 0, params)
                    operations_dict.setdefault(op_, {})[(q_ctrl, q_tgt)] = None
                    operations_qua_dict[OperationIdentifier(op_, len(params), (q_ctrl, q_tgt))] = (
                        func.apply
                    )
                    name_to_op_dict[op_] = gate_op
                    self._custom_instructions[op_] = gate_op

        for op, properties in operations_dict.items():
            target.add_instruction(name_to_op_dict[op], properties=properties)

        for flow_op_name, control_flow_op in control_flow_name_mapping.items():
            target.add_instruction(control_flow_op, name=flow_op_name)

        return target, operations_qua_dict

    def get_quam_channel(self, channel: QiskitChannel):
        """
        Convert a Qiskit Pulse channel to a QuAM channel

        Args:
            channel: The Qiskit Pulse Channel to convert

        Returns:
            The corresponding QuAM channel
        """
        try:
            return self.channel_mapping[channel]
        except KeyError:
            raise ValueError(f"Channel {channel} not in the channel mapping")

    def get_pulse_channel(self, channel: QuAMChannel):
        """
        Convert a QuAM channel to a Qiskit Pulse channel

        Args:
            channel: The QuAM channel to convert

        Returns:
            The corresponding pulse channel
        """
        return self.reverse_channel_mapping[channel]

    def meas_map(self) -> List[List[int]]:
        return self._target.concurrent_measurements

    def drive_channel(self, qubit: int):
        """
        Get the drive channel for a given qubit (should be mapped to a quantum element in configuration)
        """
        return DriveChannel(qubit)

    def measure_channel(self, qubit: int):
        return MeasureChannel(qubit)

    def acquire_channel(self, qubit: int):
        return AcquireChannel(qubit)

    def control_channel(self, qubits: Iterable[int]):
        """Return the secondary drive channel for the given qubit

        This is typically used for controlling multiqubit interactions.
        This channel is derived from other channels.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Args:
            qubits: Tuple or list of qubits of the form
                ``(control_qubit, target_qubit)``.

        Returns:
            List[ControlChannel]: The multi qubit control line.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        channels = []
        qubits = list(qubits)
        if len(qubits) != 2:
            raise ValueError("Control channel should be defined for a qubit pair")
        if self.channel_mapping is None:
            raise ValueError("Channel mapping not defined")
        for channel, element in self.channel_mapping.items():
            if isinstance(channel, ControlChannel):
                qubit_pair: QubitPair = element.parent
                qubit_control = qubit_pair.qubit_control
                qubit_target = qubit_pair.qubit_target
                q_ctrl_idx = self.qubit_dict[qubit_control.name]
                q_tgt_idx = self.qubit_dict[qubit_target.name]
                if (q_ctrl_idx, q_tgt_idx) == tuple(qubits):
                    channels.append(channel)
        if len(channels) == 0:
            raise ValueError(
                f"Control channel not found for qubit pair {qubits} in the channel mapping"
            )
        return channels

    def run(self, run_input: RunInput | List[RunInput], **options) -> QMJob:
        """
        Run a QuantumCircuit on the QOP backend (currently not supported)
        Args:
            run_input: The QuantumCircuit (or list thereof) to run on the backend. Can
            also be a Qiskit Pulse Schedule or ScheduleBlock
            options: The options for the run (can be passed as a dictionary or as keyword arguments). It
            could contain the following keys:
                - shots: The number of shots to run the circuit for (default is 1024)
                - simulate: The simulation configuration to use (if any) on the QOP
                - compiler_options: The options for the QOP compiler (if any)

        Returns:
            A QMJob object that can be used to retrieve the results of the job
        """
        from .qm_job import QMJob

        num_shots = options.get("shots", self.options.shots)
        simulate = options.get("simulate", self.options.simulate)
        compiler_options = options.get("compiler_options", self.options.compiler_options)
        skip_reset = options.get("skip_reset", self.options.skip_reset)
        memory = options.get("memory", self.options.memory)
        if not isinstance(run_input, list):
            run_input = [run_input]
        new_circuits = validate_circuits(
            run_input, should_reset=not skip_reset, check_for_params=True
        )
        num_circuits = len(new_circuits)

        self.update_calibrations()
        run_program = self.get_run_program(num_shots, new_circuits)
        qm = self.qm

        id = "pending"
        cregs_dicts = [{creg.name: creg.size for creg in qc.cregs} for qc in new_circuits]
        for i, qc in enumerate(new_circuits):
            solo_bits = [bit for bit in qc.clbits if len(qc.find_bit(bit).registers) == 0]
            if len(solo_bits) > 0:
                cregs_dicts[i][_QASM3_DUMP_LOOSE_BIT_PREFIX] = len(solo_bits)

        def result_function(qm_job: RunningQmJob) -> Result:
            results_handle = qm_job.result_handles
            results_handle.wait_for_all_values()

            # Collect all data from stream processing in the correct registers and feed
            # it to the result object
            all_data = []
            for i in range(num_circuits):
                qc_meas_data = {}
                for creg, creg_size in cregs_dicts[i].items():
                    data = results_handle.get(f"{creg}_{i}").fetch_all()["value"]
                    # time_stamps = results_handle.get(f'{creg}_{i}').fetch_all()["timestamp"]
                    bit_array = BitArray.from_samples(data, creg_size)
                    qc_meas_data[creg] = bit_array

                sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
                all_data.append(sampler_data.join_data())

            experiment_data = []
            for data in all_data:
                experiment_result = ExperimentResult(
                    shots=num_shots,
                    success=True,
                    data=ExperimentResultData(
                        data.get_counts(),
                        memory=data.get_bitstrings() if memory else None,
                    ),
                )
                experiment_data.append(experiment_result)

            result = Result(
                data=experiment_data if num_circuits > 1 else experiment_data[0],
                header={"backend_name": self.name, "job_id": qm_job.id},
            )
            return result

        job = QMJob(
            self,
            id,
            qm,
            run_program,
            simulate=simulate,
            compiler_options=compiler_options,
            result_function=result_function,
        )

        job.submit()
        return job

    def get_run_program(self, num_shots, circuits: List[QuantumCircuit]) -> Program:
        num_circuits = len(circuits)

        def process_circuit(
            qc: QuantumCircuit,
            state_int: QuaScalar[int],
            shot_var: QuaScalar[int],
            reg_streams: List[_ResultSource],
            solo_bits_stream: Optional[_ResultSource] = None,
        ):
            with for_(shot_var, 0, shot_var < num_shots, shot_var + 1):
                result = self.qiskit_to_qua_macro(qc)

                clbits_dict = {
                    creg.name: [result.result_program[creg.name][i] for i in range(creg.size)]
                    for creg in qc.cregs
                }
                num_solo_bits = len(
                    [bit for bit in circuits[0].clbits if len(qc.find_bit(bit).registers) == 0]
                )
                if num_solo_bits > 0:
                    if solo_bits_stream is None:
                        raise ValueError(
                            "Circuit contains bits without registers but no stream provided"
                        )
                    clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX] = [
                        result.result_program[f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{i}"]
                        for i in range(num_solo_bits)
                    ]
                # Save integer state to each stream

                for creg, stream in zip(qc.cregs, reg_streams):
                    for i in range(creg.size):
                        assign(state_int, state_int + (1 << i) * clbits_dict[creg.name][i])
                    save(state_int, stream)
                    assign(state_int, 0)
                if num_solo_bits > 0:
                    for i in range(num_solo_bits):
                        assign(
                            state_int,
                            state_int + (1 << i) * clbits_dict[_QASM3_DUMP_LOOSE_BIT_PREFIX][i],
                        )
                    save(state_int, solo_bits_stream)
                    assign(state_int, 0)

        with program() as prog:
            if self._init_macro:
                self._init_macro()

            shot = declare(int)
            state_int = declare(int, value=0)
            num_registers = [len(circuits[i].cregs) for i in range(num_circuits)]

            num_solo_bits = [
                len(
                    [
                        bit
                        for bit in circuits[0].clbits
                        if len(circuits[i].find_bit(bit).registers) == 0
                    ]
                )
                for i in range(num_circuits)
            ]
            regs_streams = [
                [declare_stream() for _ in range(num_cregs)] for num_cregs in num_registers
            ]
            solo_bits_stream = [declare_stream() for _ in range(num_circuits)]

            if num_circuits == 1:
                process_circuit(
                    circuits[0],
                    state_int,
                    shot,
                    regs_streams[0],
                    solo_bits_stream[0] if num_solo_bits[0] > 0 else None,
                )
            else:
                qc_var = declare(int)
                with for_(qc_var, 0, qc_var < len(circuits), qc_var + 1):
                    with switch_(qc_var):
                        for i, qc in enumerate(circuits):
                            with case_(i):
                                process_circuit(
                                    qc,
                                    state_int,
                                    shot,
                                    regs_streams[i],
                                    solo_bits_stream[i] if num_solo_bits[i] > 0 else None,
                                )

            with stream_processing():
                for i, creg_streams in enumerate(regs_streams):
                    for creg, creg_stream in zip(circuits[i].cregs, creg_streams):
                        creg_stream.save_all(f"{creg.name}_{i}")

        return prog

    def schedule_to_qua_macro(
        self,
        sched: Schedule,
        param_table: Optional[ParameterTable] = None,
        input_type: Optional[InputType] = None,
    ) -> Callable:
        """
        Convert a Qiskit Pulse Schedule to a QUA macro

        Args:
            sched: The Qiskit Pulse Schedule to convert
            param_table: The parameter table to use for the conversion of parameterized pulses to QUA variables

        Returns:
            The QUA macro corresponding to the Qiskit Pulse Schedule
        """
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
                sigParam(param, sigParam.POSITIONAL_OR_KEYWORD) for param in involved_parameters
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
                    param_table.get_parameter(param_name).assign_value(value)

            time_tracker = {channel: 0 for channel in sched.channels}

            for time, instruction in sched.instructions:
                if len(instruction.channels) > 1:
                    raise NotImplementedError("Only single channel instructions are supported")
                qiskit_channel = instruction.channels[0]

                if qiskit_channel.is_parameterized():  # Basic support for parameterized channels
                    # Filter dictionary of pulses based on provided ChannelType
                    channel_dict = {
                        channel.index: quam_channel
                        for channel, quam_channel in self.channel_mapping.items()
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
                                qiskit_channel = self.get_pulse_channel(quam_channel)
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
                    quam_channel = self.get_quam_channel(qiskit_channel)
                    if time_tracker[qiskit_channel] < time:
                        quam_channel.wait((time - time_tracker[qiskit_channel]))
                        time_tracker[qiskit_channel] = time
                    _instruction_to_qua(instruction, quam_channel, param_table)
                    time_tracker[qiskit_channel] += instruction.duration

        qua_macro.__name__ = sched.name if sched.name else "macro" + str(id(sched))
        qua_macro.__signature__ = sig
        return qua_macro

    def add_pulse_operations(
        self,
        pulse_input: Union[Schedule, ScheduleBlock, QiskitPulse],
        name: Optional[str] = None,
    ):
        """
        Add pulse operations created in Qiskit to QuAM operations mapping

        Args:
            pulse_input: The pulse input to add to the QuAM operations mapping (can be a Schedule, ScheduleBlock or Pulse)
            name: An optional name to refer to the pulse operations to be added to the QuAM operations mapping. If
            a Schedule or ScheduleBlock is provided, all pulse operations are named as "{name}_{i}" where i is the number
            of the pulse operation in the schedule. If a Pulse is provided, it is named as "{name}".
        """
        if isinstance(pulse_input, QiskitPulse):
            pulse_input = Schedule(Play(pulse_input, DriveChannel(0)))

        pulse_input = validate_schedule(pulse_input)

        # Update QuAM with additional custom pulses
        for idx, (time, instruction) in enumerate(
            pulse_input.filter(instruction_types=[Play]).instructions
        ):
            instruction: Play
            pulse, channel = instruction.pulse, instruction.channel
            if not isinstance(pulse, (SymbolicPulse, Waveform)):
                raise ValueError("Only SymbolicPulse and Waveform pulses are supported")

            pulse_name = pulse.name
            if (
                not channel.is_parameterized()
                and pulse_name in self.get_quam_channel(channel).operations
            ):
                pulse_name += str(pulse.id)
                pulse.name = pulse_name

            # Check if pulse fits QOP constraints
            if pulse.duration < 16:
                raise ValueError("Pulse duration must be at least 16 ns")
            elif pulse.duration % 4 != 0:
                raise ValueError("Pulse duration must be a multiple of 4 ns")
            if pulse.name is None:
                if name is not None:
                    pulse.name = f"{name}_{idx}"
                else:
                    pulse.name = f"qiskit_pulse_{id(pulse)}"
            quam_pulse = QuAMQiskitPulse(pulse)
            if quam_pulse.is_compile_time_parameterized():
                raise ValueError(
                    "Pulse contains unassigned parameters that cannot be adjusted in real-time"
                )

            if channel.is_parameterized():  # Add pulse to each channel of same type
                for ch in filter(
                    lambda x: isinstance(x, type(channel)),
                    self.channel_mapping.keys(),
                ):
                    self.get_quam_channel(ch).operations[pulse.name] = QuAMQiskitPulse(pulse)
            else:
                self.get_quam_channel(channel).operations[pulse.name] = QuAMQiskitPulse(pulse)

    def update_calibrations(
        self,
        qc: Optional[QuantumCircuit | List[QuantumCircuit]] = None,
        input_type: Optional[InputType] = None,
    ):
        """
        This method updates the QuAM with the custom calibrations of the QuantumCircuit (if any)
        and adds the corresponding operations to the QUA operations mapping for the OQC compiler.
        This method should be called before opening the QuantumMachine instance (i.e. before generating the
        configuration through QuAM) as it modifies the QuAM configuration.
        It also looks at the Target object and checks if new operations are added to the target. If
        so, it adds them to the QUA operations mapping for the OQC compiler.

        Args:
            qc: The QuantumCircuit to update the calibrations from. If None, only the target is checked
            input_type: The input type for the parameter table (if any). Relevant only if the parameter table is
            not already defined and present in qc metadata
        """
        # Check the target object for new operations
        for op_name, op_properties in self.target.items():
            gate_set = list(set(key.name for key in self._operation_mapping_QUA.keys())) + list(
                CONTROL_FLOW_OP_NAMES
            )
            if op_name not in gate_set:
                for qubits, properties in op_properties.items():
                    if properties is None:
                        raise ValueError(
                            f"Operation {op_name} has no properties defined in the target,"
                            f"hence cannot be added to the QUA operations mapping"
                        )
                    elif isinstance(properties, QMInstructionProperties):
                        if properties.qua_pulse_macro is None:
                            raise ValueError(
                                f"Operation {op_name} has no QUA macro defined in the target,"
                                f"hence cannot be added to the QUA operations mapping"
                            )
                        sched = properties.qua_pulse_macro
                        sig = Signature.from_callable(sched)
                        positional_params = [
                            param
                            for param in sig.parameters.values()
                            if param.kind
                            in (
                                sigParam.POSITIONAL_OR_KEYWORD,
                                sigParam.POSITIONAL_ONLY,
                            )
                        ]
                        num_params = len(positional_params)
                        op_id = OperationIdentifier(
                            op_name,
                            num_params,
                            qubits,
                        )
                        self._operation_mapping_QUA[op_id] = sched

                    elif isinstance(properties, InstructionProperties):
                        if properties.calibration is None:
                            raise ValueError(
                                f"Operation {op_name} has no calibration defined in the target,"
                                f"hence cannot be added to the QUA operations mapping"
                            )
                        sched = validate_schedule(properties.calibration)
                        num_params = len(sched.parameters)
                        if num_params > 0:
                            param_table = sched.metadata.get(
                                "qua",
                                ParameterTable.from_qiskit(
                                    sched,
                                    input_type=input_type,
                                    name=sched.name + "_param_table",
                                ),
                            )

                        else:
                            param_table = None
                        self._operation_mapping_QUA[
                            OperationIdentifier(
                                op_name,
                                num_params,
                                qubits,
                            )
                        ] = self.schedule_to_qua_macro(sched, param_table)

        if qc is not None:
            if not isinstance(qc, QuantumCircuit):
                raise ValueError("qc should be a QuantumCircuit")
            if not hasattr(qc, "calibrations"):
                raise ValueError("qc should have calibrations")
            if qc.parameters or qc.iter_vars():
                param_table = qc.metadata.get(
                    "parameter_table",
                    ParameterTable.from_qiskit(
                        qc, input_type=input_type, name=qc.name + "_param_table"
                    ),
                )
            else:
                param_table = None

            if hasattr(qc, "calibrations") and qc.calibrations:  # Check for custom calibrations
                for gate_name, cal_info in qc.calibrations.items():
                    if (
                        gate_name not in self._oq3_custom_gates
                    ):  # Make it a basis gate for OQ compiler
                        self._oq3_custom_gates.append(gate_name)
                    for (qubits, parameters), schedule in cal_info.items():
                        schedule = validate_schedule(
                            schedule
                        )  # Check that schedule has fixed duration

                        # Convert type of parameters to int if required (for switch case over channels)
                        if param_table is not None:
                            param_table = handle_parameterized_channel(schedule, param_table)

                        self._operation_mapping_QUA[
                            OperationIdentifier(
                                gate_name,
                                len(parameters),
                                qubits,
                            )
                        ] = self.schedule_to_qua_macro(schedule, param_table)

                        self.add_pulse_operations(schedule, name=schedule.name)

    def quantum_circuit_to_qua(
        self,
        qc: QuantumCircuit,
        param_table: Optional[ParameterTable | List[ParameterTable | Parameter]] = None,
    ):
        """
        Convert a QuantumCircuit to a QUA program (can be called within an existing QUA program or to generate a
        program for the circuit)

        Args:
            qc: The QuantumCircuit to convert
            param_table: The parameter table to use for the conversion of parameterized instructions to QUA variables
                        Should be provided if the QuantumCircuit contains real-time variables or symbolic Parameters
                         to be cast as real-time parameters (typically amp, phase, frequency or duration parameters)
                         and this function is called within a QUA program

        Returns:
            Compilation result of the QuantumCircuit to QUA
        """
        # if qc.parameters and param_table is None:
        #     raise ValueError(
        #         "QuantumCircuit contains parameters but no parameter table provided"
        #     )
        basis_gates = list(
            set(self._oq3_custom_gates + list(self.target.operation_names))
            - set(oq3_keyword_instructions)
        )
        # Check if all custom calibrations are in the oq3 basis gates
        for gate_name in qc.calibrations.keys():
            if gate_name not in basis_gates:
                raise ValueError(
                    f"Custom calibration {gate_name} not in basis gates {basis_gates}",
                    f"Run update_calibrations() before compiling the circuit",
                )
        exporter = Exporter(includes=(), basis_gates=basis_gates, disable_constants=True)
        open_qasm_code = exporter.dumps(qc)
        open_qasm_code = "\n".join(
            line
            for line in open_qasm_code.splitlines()
            if not line.strip().startswith(("barrier",))
        )
        inputs = None
        if param_table is not None:
            inputs = {}
            if isinstance(param_table, (ParameterTable, Parameter)):
                param_table = [param_table]
            for table in param_table:
                if not table.is_declared:
                    if isinstance(table, ParameterTable):
                        table.declare_variables(pause_program=False)
                    else:
                        table.declare_variable(pause_program=False)
                variables = (
                    table.variables_dict
                    if isinstance(table, ParameterTable)
                    else {table.name: table.var}
                )
                inputs.update(variables)

        result = self.compiler.compile(
            open_qasm_code,
            compilation_name=f"{qc.name}_qua",
            inputs=inputs,
        )
        return result

    def qiskit_to_qua_macro(
        self,
        qc: RunInput,
        input_type: Optional[InputType] = None,
    ) -> CompilationResult | Program | Callable[..., Any]:
        """
        Convert given input into a QUA program
        """

        if qc.parameters:  # Initialize the parameter table
            parameter_table = ParameterTable.from_qiskit(qc, input_type=input_type)
            qc.metadata["parameter_table"] = parameter_table
        else:
            parameter_table = None
        if isinstance(qc, QuantumCircuit):
            return self.quantum_circuit_to_qua(qc, parameter_table)
        elif isinstance(qc, (ScheduleBlock, Schedule)):  # Convert to Schedule first
            schedule = validate_schedule(qc)

            return self.schedule_to_qua_macro(schedule, parameter_table)
        else:
            raise ValueError(f"Unsupported input {qc}")

    @property
    def compiler(self) -> Compiler:
        """
        The OpenQASM to QUA compiler.
        """
        return Compiler(
            hardware_config=HardwareConfig(
                quantum_operations_db=self._operation_mapping_QUA,
                physical_qubits=self.qubit_mapping,
            )
        )

    def connect(self) -> QuantumMachinesManager:
        """
        Connect to the Quantum Machines Manager
        """
        return self.machine.connect()

    def generate_config(self) -> DictQuaConfig:
        """
        Generate the configuration for the Quantum Machine
        """
        return self.machine.generate_config()

    @property
    def init_macro(self) -> Optional[Callable]:
        """
        The macro to be called at the beginning of the QUA program
        """
        return self._init_macro


class FluxTunableTransmonBackend(QMBackend):

    def __init__(
        self,
        machine: Quam,
    ):
        """
        Initialize the QM backend for the Flux-Tunable Transmon based QuAM

        Args:
            machine: The QuAM instance
            channel_mapping: Optional mapping of Qiskit Pulse Channels to QuAM Channels.
                             This mapping enables the conversion of Qiskit schedules into parametric QUA macros.
        """
        if not hasattr(machine, "qubits") or not hasattr(machine, "qubit_pairs"):
            raise ValueError(
                "Invalid QuAM instance provided, should have qubits and qubit_pairs attributes"
            )
        drive_channel_mapping = {
            DriveChannel(i): qubit.xy for i, qubit in enumerate(machine.qubits.values())
        }
        flux_channel_mapping = {
            FluxChannel(i): qubit.z for i, qubit in enumerate(machine.qubits.values())
        }
        readout_channel_mapping = {
            MeasureChannel(i): qubit.resonator for i, qubit in enumerate(machine.qubits.values())
        }
        control_channel_mapping = {
            ControlChannel(i): qubit_pair.coupler
            for i, qubit_pair in enumerate(machine.qubit_pairs.values())
        }
        channel_mapping = {
            **drive_channel_mapping,
            **flux_channel_mapping,
            **control_channel_mapping,
            **readout_channel_mapping,
        }
        super().__init__(
            machine,
            channel_mapping=channel_mapping,
            init_macro=machine.apply_all_flux_to_joint_idle,
        )

    @property
    def qubit_mapping(self) -> QubitsMapping:
        """
        Retrieve the qubit to quantum elements mapping for the backend.
        """
        return {
            i: (qubit.xy.name, qubit.z.name, qubit.resonator.name)
            for i, qubit in enumerate(self.machine.qubits.values())
        }

    @property
    def meas_map(self) -> List[List[int]]:
        """
        Retrieve the measurement map for the backend.
        """
        return [[i] for i in range(len(self.machine.qubits))]

    def flux_channel(self, qubit: int):
        """
        Retrieve the flux channel for the given qubit.
        """
        return FluxChannel(qubit)


def qua_declaration(n_qubits, readout_elements):
    """
    Macro to declare the necessary QUA variables

    :param n_qubits: Number of qubits used in this experiment
    :return:
    """
    I, Q = [[declare(fixed) for _ in range(n_qubits)] for _ in range(2)]
    I_st, Q_st = [[declare_stream() for _ in range(n_qubits)] for _ in range(2)]
    # Workaround to manually assign the results variables to the readout elements
    for i in range(n_qubits):
        assign_variables_to_element(readout_elements[i], I[i], Q[i])
    return I, I_st, Q, Q_st
