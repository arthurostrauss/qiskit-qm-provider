from typing import Union, Literal, Optional
from quam.components.macro import QubitMacro
from quam.components import Qubit
from quam.core import quam_dataclass
from quam.components.pulses import Pulse, ReadoutPulse
from quam.utils.qua_types import QuaVariableBool, StreamType

from qm.qua import declare, fixed, save, assign, wait
from quam_builder.architecture.superconducting.components.readout_resonator import ReadoutResonatorIQ
from quam_builder.architecture.superconducting.qubit import AnyTransmon

def get_pulse_name(pulse: Pulse) -> str:
    """
    Get the name of the pulse. If the pulse has an id, return it.
    """
    if pulse.id is not None:
        return pulse.id
    elif pulse.parent is not None:
        return pulse.parent.get_attr_name(pulse)
    else:
        raise AttributeError(
            f"Cannot infer id of {pulse} because it is not attached to a parent"
        )

def get_pulse(pulse: Pulse | str, qubit: Qubit | None = None) -> Pulse:
    if isinstance(pulse, Pulse):
        return pulse
    elif qubit is not None:
        return qubit.get_pulse(pulse)
    else:
        raise ValueError(f"Cannot get pulse {pulse} because qubit is not provided")

@quam_dataclass
class MeasureMacro(QubitMacro):
    """
    Macro for measuring a qubit.
    """
    pulse: Union[ReadoutPulse, str]="readout"

    def apply(self, **kwargs) -> QuaVariableBool:

        pulse = get_pulse(self.pulse, self.qubit)
        resonator: ReadoutResonatorIQ = self.qubit.resonator

        qua_vars = kwargs.get("qua_vars", (declare(fixed), declare(fixed)))
        I, Q = qua_vars
        state: QuaVariableBool = kwargs.get("state", declare(bool))
        stream_I: Optional[StreamType] = kwargs.get("stream_I", None)
        stream_Q: Optional[StreamType] = kwargs.get("stream_Q", None)

        resonator.measure(get_pulse_name(pulse), qua_vars=(I, Q))
        wait(resonator.depletion_time // 4, resonator.name)
        assign(state, I > pulse.threshold)
        if stream_I is not None:
            save(I, stream_I)
        if stream_Q is not None:
            save(Q, stream_Q)

        return state
    
    @property
    def inferred_duration(self) -> float:
        readout_pulse: ReadoutPulse = self.pulse if isinstance(self.pulse, ReadoutPulse) else get_pulse(self.pulse, self.qubit)
        return readout_pulse.length * 1e-9

@quam_dataclass
class ResetMacro(QubitMacro):
    """
    Macro for resetting a qubit.
    """
    reset_type: Literal["thermalize", "active", "active_gef"] = "active"
    pi_pulse: Union[Pulse, str]= "x180"
    readout_pulse: Union[ReadoutPulse, str]= "readout"
    pi_12_pulse: Optional[Union[Pulse, str]]= None
    max_attempts: int= 5

    def apply(self, **kwargs) -> None:

        qubit: AnyTransmon = self.qubit
        pi_pulse = get_pulse(self.pi_pulse, self.qubit)
        pi_12_pulse = get_pulse(self.pi_12_pulse, self.qubit) if self.pi_12_pulse is not None else None
        readout_pulse = get_pulse(self.readout_pulse, self.qubit)
        if self.reset_type == "thermalize":
            qubit.reset_qubit_thermal()
        elif self.reset_type == "active":
            qubit.reset_qubit_active(kwargs.get("save_qua_var", None), pi_pulse_name=get_pulse_name(pi_pulse), readout_pulse_name=get_pulse_name(readout_pulse), max_attempts=self.max_attempts)
        elif self.reset_type == "active_gef":
            if pi_12_pulse is None:
                raise ValueError("pi_12_pulse is required for active_gef reset")
            qubit.reset_qubit_active_gef(readout_pulse_name=get_pulse_name(readout_pulse), pi_01_pulse_name=get_pulse_name(pi_pulse), pi_12_pulse_name=get_pulse_name(pi_pulse))

       
    @property
    def inferred_duration(self) -> float:
        """
        This property is used to get the duration of the reset macro (in seconds).
        We provide here a worst case estimate of the duration for the case where the reset is active.
        For the case where the reset is thermalize, we return the thermalization time of the qubit.
        """
        if self.reset_type == "active":
            pi_pulse_duration = get_pulse(self.pi_pulse, self.qubit).length
            readout_pulse_duration = get_pulse(self.readout_pulse, self.qubit).length
            return (pi_pulse_duration + readout_pulse_duration) * self.max_attempts * 1e-9 # convert to seconds
        elif self.reset_type == "thermalize":
            return self.qubit.thermalization_time * 1e-9 # convert to seconds
        else: # active_gef
            raise ValueError(f"Reset type {self.reset_type} is not supported")
    
@quam_dataclass
class VirtualZMacro(QubitMacro):
    """
    Macro for applying a virtual Z gate to a qubit.
    """

    def apply(self, angle: float, **kwargs) -> None:
        qubit: AnyTransmon = self.qubit
        qubit.xy.frame_rotation(-angle)

    def __post_init__(self) -> None:
        self.fidelity = 1.0  # Virtual Z gate is assumed to be perfect

    @property
    def inferred_duration(self) -> float:
        return 0.0 # Virtual Z gate is assumed to be instantaneous

@quam_dataclass
class DelayMacro(QubitMacro):
    """
    Macro for delaying a qubit.
    """

    def apply(self, duration: int, **kwargs) -> None:
        qubit: AnyTransmon = self.qubit
        qubit.wait(duration)

@quam_dataclass
class IdMacro(QubitMacro):
    """
    Macro for applying an identity operation to a qubit.
    In QUA, we assimilate it to an align statement across all the channels of the qubit.
    """
    
    def apply(self, **kwargs) -> None:
        qubit: AnyTransmon = self.qubit
        qubit.align()
        
    @property
    def inferred_duration(self) -> float:
        return 0.0 # Identity operation is assumed to be instantaneous

    def __post_init__(self) -> None:
        self.fidelity = 1.0 # Identity operation is assumed to be perfect

