from __future__ import annotations

import datetime
import warnings
from copy import deepcopy
from typing import Iterable, List, Dict, Optional, Callable, Union, Tuple, Any, TYPE_CHECKING
from inspect import Signature, Parameter as sigParam

import numpy as np
from qiskit.circuit import (
    QuantumCircuit,
    Parameter as QiskitParameter,
    Instruction,
)
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.primitives import BitArray, SamplerPubResult, DataBin
from qiskit.providers import BackendV2 as Backend, QubitProperties, Options

from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData, MeasLevel, MeasReturnType

from qiskit.transpiler import Target, InstructionProperties, CouplingMap
from qiskit.qasm3 import Exporter
from qm.jobs.running_qm_job import RunningQmJob

# QUA and Quam imports
from qm import (
    QuantumMachinesManager,
    Program,
    DictQuaConfig,
    QuantumMachine,
    StreamingResultFetcher,
)
from quam.components import Channel as QuAMChannel, QubitPair, Qubit

# OpenQASM3 to QUA compiler
from oqc import (
    Compiler,
    HardwareConfig,
    OperationIdentifier,
    QubitsMapping,
    CompilationResult,
)

if TYPE_CHECKING:
    from quam_libs.cloud_infrastructure import (
        CloudQuantumMachine,
        CloudQuantumMachinesManager,
        CloudJob,
        CloudResultHandles,
    )
    from iqcc_cloud_client import IQCC_Cloud

# Helper modules
from ..parameter_table import ParameterTable, InputType, Parameter
from .backend_utils import (
    validate_machine,
    look_for_standard_op,
    get_extended_gate_name_mapping,
    control_flow_name_mapping,
    _QASM3_DUMP_LOOSE_BIT_PREFIX,
    oq3_keyword_instructions,
    validate_circuits,
)
from .qm_instruction_properties import QMInstructionProperties

if TYPE_CHECKING:
    from qiskit_qm_provider.job.qm_job import QMJob, IQCCJob
    from quam_builder.architecture.superconducting.qpu.base_quam import BaseQuam as Quam
__all__ = ["QMBackend", "QISKIT_PULSE_AVAILABLE"]

try:  # Importing Qiskit Pulse components
    from qiskit.pulse import (
        DriveChannel,
        MeasureChannel,
        AcquireChannel,
        ControlChannel,
        Schedule,
        ScheduleBlock,
        Play,
        Waveform,
        SymbolicPulse,
    )
    from qiskit.pulse.channels import Channel as QiskitChannel
    from qiskit.pulse.library import Pulse as QiskitPulse

    QISKIT_PULSE_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Qiskit Pulse is not available, some features of the QM backend will not be available",
        ImportWarning,
    )
    QISKIT_PULSE_AVAILABLE = False
    QiskitChannel = DriveChannel = MeasureChannel = AcquireChannel = ControlChannel = Schedule = (
        ScheduleBlock
    ) = Play = Waveform = SymbolicPulse = None


def requires_qiskit_pulse(func):
    """
    Decorator to check if Qiskit Pulse is available before executing a function.
    """

    def wrapper(*args, **kwargs):
        if not QISKIT_PULSE_AVAILABLE:
            raise ImportError(
                "Current Qiskit version does not have Qiskit Pulse, lower it to 1.x to use this feature."
            )
        return func(*args, **kwargs)

    return wrapper


class QMBackend(Backend):
    def __init__(
        self,
        machine: Quam,
        channel_mapping: Optional[Dict[QiskitChannel, QuAMChannel]] = None,
        init_macro: Optional[Callable] = None,
        qmm: Optional[QuantumMachinesManager] = None,
        name: Optional[str] = None,
        **fields,
    ):
        """
        Initialize the QM backend
        Args:
            machine: The Quam instance
            channel_mapping: Optional mapping of Qiskit Pulse Channels (e.g. DriveChannel, ControlChannel)
                             to QuAM Channels. This mapping enables the conversion of Qiskit Pulse schedules
                            into parametric QUA macros. Note: This requires Qiskit to be of version < 2.0 to work.
                            Additionally, the Schedules created must have deterministic durations at this point.
            init_macro: Optional macro to be called at the beginning of the QUA program
            qmm: Optional QuantumMachinesManager instance. If not provided, inferred from the machine
            name: Optional name of the backend
            fields: kwargs for the values to use to override the default
                options

        Raises:
            AttributeError: If a field is specified that's outside the backend's
                options



        """

        Backend.__init__(self, name="QMBackend" if name is None else name, **fields)

        self._custom_instructions = {}
        self.machine = validate_machine(machine)
        self._qmm: Optional[QuantumMachinesManager] = qmm
        self._qm: Optional[QuantumMachine] = None
        self.channel_mapping: Dict[QiskitChannel, QuAMChannel] = channel_mapping
        self.reverse_channel_mapping: Dict[QuAMChannel, QiskitChannel] = (
            {v: k for k, v in channel_mapping.items()} if channel_mapping is not None else {}
        )
        self._qubit_dict = {qubit.name: i for i, qubit in enumerate(machine.active_qubits)}
        self._qubit_pair_dict = {
            qubit_pair.name: (
                self._qubit_dict[qubit_pair.qubit_control.name],
                self._qubit_dict[qubit_pair.qubit_target.name],
            )
            for qubit_pair in machine.active_qubit_pairs
        }

        self._target, self._ref_operation_mapping_QUA, self._coupling_map = self._populate_target(
            machine
        )
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
    def qubit_pair_dict(self):
        """
        Get the qubit pair dictionary for the backend
        """
        return self._qubit_pair_dict

    def get_qubit(self, qubit: int | str) -> Qubit:
        """
        Get the Qubit object corresponding to the given qubit index or name
        Args:
            qubit: The qubit index or name

        Returns:
            The Qubit object corresponding to the given qubit index or name
        """
        if isinstance(qubit, int):
            return self.machine.active_qubits[qubit]
        elif isinstance(qubit, str):
            return self.machine.active_qubits[self.qubit_dict[qubit]]
        else:
            raise ValueError("Qubit should be an integer index or a string name")

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
    def qmm(self) -> Union[QuantumMachinesManager, CloudQuantumMachinesManager, IQCC_Cloud]:
        """
        Returns the QuantumMachinesManager instance. This is a property that reopens a QuantumMachinesManager each time
        it is called for the underlying configuration might have changed between two calls
        """
        if self._qmm is None:
            self._qmm = self.machine.connect()
        return self._qmm

    @qmm.setter
    def qmm(self, qmm: Union[QuantumMachinesManager, CloudQuantumMachinesManager]):
        """
        Set the QuantumMachinesManager instance. This is a property that reopens a QuantumMachinesManager each time
        it is called for the underlying configuration might have changed between two calls
        """
        self._qmm = qmm

    @property
    def qm(self) -> Union[QuantumMachine, CloudQuantumMachine, IQCC_Cloud]:
        """
        Returns the QuantumMachine instance. This is a property that reopens a QuantumMachine each time
        it is called for the underlying configuration might have changed between two calls
        """
        try:
            from quam_libs.cloud_infrastructure import CloudQuantumMachinesManager
        except ImportError:
            CloudQuantumMachinesManager = None
        if self._qm is None:
            if isinstance(self.qmm, (QuantumMachinesManager, CloudQuantumMachinesManager)):
                self._qm = self.qmm.open_qm(self.qm_config, close_other_machines=True)
            else:
                self._qm = self.qmm

        return self._qm

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
            meas_level=MeasLevel.CLASSIFIED,
            meas_return=MeasReturnType.AVERAGE,
            timeout=60,
        )

    def _populate_target(
        self, machine: Quam
    ) -> Tuple[Target, Dict[OperationIdentifier, Callable], CouplingMap]:
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
                QubitProperties(t1=qubit.T1, t2=qubit.T2echo, frequency=qubit.f_01)
                for qubit in machine.active_qubits
            ],
        )

        operations_dict = {}
        operations_qua_dict = {}
        name_to_op_dict = {}
        coupling_map = []

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
            coupling_map.append([q_ctrl, q_tgt])
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

        return target, operations_qua_dict, CouplingMap(coupling_map)

    @requires_qiskit_pulse
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

    @requires_qiskit_pulse
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

    @requires_qiskit_pulse
    def drive_channel(self, qubit: int):
        """
        Get the drive channel for a given qubit (should be mapped to a quantum element in configuration)
        """
        return DriveChannel(qubit)

    @requires_qiskit_pulse
    def measure_channel(self, qubit: int):
        return MeasureChannel(qubit)

    @requires_qiskit_pulse
    def acquire_channel(self, qubit: int):
        return AcquireChannel(qubit)

    @requires_qiskit_pulse
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

    def run(self, run_input: QuantumCircuit | List[QuantumCircuit], **options) -> QMJob:
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
        from ..job.qm_job import QMJob, IQCCJob
        from ..job.qua_programs import get_run_program

        options_ = deepcopy(self.options.__dict__)
        options_.update(options)
        num_shots = options.get("shots", self.options.shots)
        skip_reset = options.get("skip_reset", self.options.skip_reset)
        memory = options.get("memory", self.options.memory)
        meas_level = options.get("meas_level", self.options.meas_level)
        meas_return = options.get("meas_return", self.options.meas_return)
        if not isinstance(run_input, list):
            run_input = [run_input]
        new_circuits = validate_circuits(
            run_input, should_reset=not skip_reset, check_for_params=True
        )
        num_circuits = len(new_circuits)
        self.update_target()
        if QISKIT_PULSE_AVAILABLE:
            for qc in new_circuits:
                self.update_calibrations(qc)

        run_program = get_run_program(self, num_shots, new_circuits)
        qm = self.qm

        id = "pending"
        cregs_dicts = [{creg.name: creg.size for creg in qc.cregs} for qc in new_circuits]
        for i, qc in enumerate(new_circuits):
            solo_bits = [bit for bit in qc.clbits if len(qc.find_bit(bit).registers) == 0]
            if len(solo_bits) > 0:
                cregs_dicts[i][_QASM3_DUMP_LOOSE_BIT_PREFIX] = len(solo_bits)

        def result_function(qm_job: RunningQmJob | List[RunningQmJob] | CloudJob | Dict) -> Result:
            is_job_list = isinstance(qm_job, list)
            if is_job_list:
                results_handle = [job.result_handles for job in qm_job]
                for handle in results_handle:
                    handle.wait_for_all_values()
            elif isinstance(qm_job, (RunningQmJob, CloudJob)):
                results_handle = qm_job.result_handles
                results_handle.wait_for_all_values()
            else:
                results_handle = qm_job["result"]

            # Collect all data from stream processing in the correct registers and feed
            # it to the result object
            all_data = []
            for i in range(num_circuits):
                qc_meas_data = {}
                for creg, creg_size in cregs_dicts[i].items():
                    if is_job_list:
                        data = (
                            np.array(results_handle[i].get(f"{creg}_{i}").fetch_all()["value"])
                            .flatten()
                            .tolist()
                        )
                    elif isinstance(results_handle, (StreamingResultFetcher, CloudResultHandles)):
                        data = (
                            np.array(results_handle.get(f"{creg}_{i}").fetch_all()["value"])
                            .flatten()
                            .tolist()
                        )
                    else:
                        data = np.array(results_handle.get(f"{creg}_{i}")).flatten().tolist()
                    # time_stamps = results_handle.get(f'{creg}_{i}').fetch_all()["timestamp"]
                    if meas_level == MeasLevel.CLASSIFIED:
                        bit_array = BitArray.from_samples(data, creg_size)
                        qc_meas_data[creg] = bit_array
                    elif meas_level == MeasLevel.KERNELED:
                        if meas_return == MeasReturnType.SINGLE:
                            qc_meas_data[creg] = np.array(
                                [d[0] + 1j * d[1] for d in data], dtype=complex
                            )
                        elif meas_return == MeasReturnType.AVERAGE:
                            qc_meas_data[creg] = np.mean([d[0] + 1j * d[1] for d in data])

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
                    meas_level=meas_level,
                    meas_return=meas_return,
                    status=qm_job.status if isinstance(qm_job, RunningQmJob) else "done",
                )
                experiment_data.append(experiment_result)

            result = Result(
                results=experiment_data if num_circuits > 1 else experiment_data[0],
                backend_name=self.name,
                job_id=qm_job.id if hasattr(qm_job, "id") else "unknown",
                backend_version=2,
                qobj_id=None,
                success=True,
                date=datetime.datetime.now().isoformat(),
            )
            return result

        job_obj = QMJob if isinstance(self.qmm, QuantumMachinesManager) else IQCCJob
        job = job_obj(
            self,
            id,
            qm,
            run_program,
            result_function=result_function,
            config=self.qm_config,
            **options_,
        )

        job.submit()
        self._operation_mapping_QUA = self._ref_operation_mapping_QUA.copy()
        return job

    @requires_qiskit_pulse
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
            input_type: The input type to use for the conversion of parameterized pulses to QUA variables.
                Should be specified only if the schedule is parameterized and the parameter table is not provided.

        Returns:
            The QUA macro corresponding to the Qiskit Pulse Schedule
        """

        from ..pulse import schedule_to_qua_macro

        return schedule_to_qua_macro(self, sched, param_table, input_type)

    @requires_qiskit_pulse
    def add_pulse_operations(
        self,
        pulse_input: Union[Schedule, ScheduleBlock],
        name: Optional[str] = None,
    ):
        """
        Add pulse operations created in Qiskit to QuAM operations mapping

        Args:
            pulse_input: The pulse input to add to the QuAM operations mapping (can be a Schedule, ScheduleBlock)
            name: An optional name to refer to the pulse operations to be added to the QuAM operations mapping. If
            a Schedule or ScheduleBlock is provided, all pulse operations are named as "{name}_{i}" where i is the number
            of the pulse operation in the schedule. If a Pulse is provided, it is named as "{name}".
        """
        from ..pulse import validate_schedule, QuAMQiskitPulse

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

    def update_compiler_from_target(
        self,
        input_type: Optional[InputType] = None,
    ):
        """
        Update the target with the operations defined in the target object.
        :param input_type: Input type to use for the conversion of parameterized instructions to QUA variables.
        :return:
        """
        # Check the target object for new operations
        for op_name, op_properties in self.target.items():
            gate_set = list(set(key.name for key in self._ref_operation_mapping_QUA.keys())) + list(
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
                        self._ref_operation_mapping_QUA[op_id] = sched

                    elif isinstance(properties, InstructionProperties) and hasattr(
                        properties, "calibration"
                    ):
                        from ..pulse.pulse_support_utils import validate_schedule

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
                            if isinstance(param_table, Dict):
                                if len(param_table) == 1:
                                    param_table = list(param_table.values())[0]
                                else:
                                    param_table = ParameterTable.from_other_tables(
                                        list(param_table.values())
                                    )

                        else:
                            param_table = None
                        self._ref_operation_mapping_QUA[
                            OperationIdentifier(
                                op_name,
                                num_params,
                                qubits,
                            )
                        ] = self.schedule_to_qua_macro(sched, param_table)

        self._operation_mapping_QUA = self._ref_operation_mapping_QUA.copy()

    def update_target(self):
        """
        Update the target with the operations defined in the machine macros (if new macros were added)
        """
        self._target, self._ref_operation_mapping_QUA, self._coupling_map = self._populate_target(
            self.machine
        )

    @requires_qiskit_pulse
    def update_calibrations(self, qc: QuantumCircuit, input_type: Optional[InputType] = None):
        """
        Update the QUA operations mapping with the calibrations defined in the QuantumCircuit.
        Works only with Qiskit version below 2.0 (i.e. with Qiskit Pulse).
        :param qc: QuantumCircuit to update the calibrations from.
        :param input_type: Input type to use for the conversion of parameterized instructions to QUA variables.
        :return:
        """
        if hasattr(qc, "calibrations") and qc.calibrations:  # Check for custom calibrations
            from ..pulse.pulse_support_utils import validate_schedule, handle_parameterized_channel

            if qc.parameters or qc.iter_vars():
                param_table = qc.metadata.get(
                    "qua",
                    ParameterTable.from_qiskit(
                        qc, input_type=input_type, name=qc.name + "_param_table"
                    ),
                )
                if isinstance(param_table, Dict):
                    if len(param_table) == 1:
                        param_table = list(param_table.values())[0]
                    else:
                        param_table = ParameterTable.from_other_tables(list(param_table.values()))

            else:
                param_table = None

            for gate_name, cal_info in qc.calibrations.items():
                if gate_name not in self._oq3_custom_gates:  # Make it a basis gate for OQ compiler
                    self._oq3_custom_gates.append(gate_name)
                for (qubits, parameters), schedule in cal_info.items():
                    schedule = validate_schedule(schedule)  # Check that schedule has fixed duration

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
        else:
            warnings.warn("No calibrations found in the QuantumCircuit", UserWarning)

    def quantum_circuit_to_qua(
        self,
        qc: QuantumCircuit,
        param_table: Optional[ParameterTable | List[ParameterTable | Parameter]] = None,
    ) -> CompilationResult:
        """
        Convert a QuantumCircuit to a QUA program (can be called within an existing QUA program or to generate a
        program for the circuit).
        If executed outside of a QUA program scope, the resulting QUA program can be accessed through:
        prog = backend.quantum_circuit_to_qua(qc).result_program.dsl_program

        Args:
            qc: The QuantumCircuit to convert
            param_table: The parameter table to use for the conversion of parameterized instructions to QUA variables
                        Should be provided if the QuantumCircuit contains real-time variables or symbolic Parameters
                         to be cast as real-time parameters (typically amp, phase, frequency or duration parameters)
                         and this function is called within a QUA program

        Returns:
            Compilation result of the QuantumCircuit to QUA
        """

        basis_gates = self.oqc_basis_gates
        # Check if all custom calibrations are in the oq3 basis gates
        if hasattr(qc, "calibrations") and qc.calibrations:
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
        qc: QuantumCircuit,
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
            from ..pulse import validate_schedule

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

    @init_macro.setter
    def init_macro(self, macro: Callable):
        """
        Set the macro to be called at the beginning of the QUA program
        """
        if not callable(macro):
            raise ValueError("Init macro must be a callable")
        self._init_macro = macro

    @property
    def qubits(self) -> List[Qubit]:
        """
        Retrieve the list of active qubits of the machine
        """
        return self.machine.active_qubits

    @property
    def qubit_pairs(self) -> List[QubitPair]:
        """
        Retrieve the list of active qubit pairs of the machine
        """
        return self.machine.active_qubit_pairs

    @property
    def oqc_basis_gates(self) -> List[str]:
        """
        Retrieve the list of OpenQASM 3 basis gates supported by the backend
        """
        basis_gates = list(
            set(self._oq3_custom_gates + list(self.target.operation_names))
            - set(oq3_keyword_instructions)
        )
        return basis_gates

    @property
    def oq3_exporter(self) -> Exporter:
        """
        Retrieve the OpenQASM 3 exporter for the backend
        """
        return Exporter(
            includes=(),
            basis_gates=self.oqc_basis_gates,
            disable_constants=True,
        )
