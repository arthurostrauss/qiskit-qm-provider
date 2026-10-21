"""Build a dummy QUAM state from QMM controller discovery for testing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Type

from qm import QuantumMachinesManager
from quam.components.ports import FEMPortsContainer
from qualang_tools.wirer.connectivity.wiring_spec import WiringLineType

from quam_builder.architecture.superconducting.qpu import AnyQuam
from quam_builder.architecture.superconducting.qubit import FluxTunableTransmon
from quam_builder.builder.superconducting.add_default_pulses import (
    add_DragCosine_pulses,
    add_default_transmon_pair_pulses,
    add_default_transmon_pulses,
)
from quam_builder.builder.superconducting.build_quam import (
    add_external_mixers,
    add_ports,
    add_transmons,
)
from quam_builder.builder.superconducting.add_default_macros import (
    add_default_transmon_macros,
    add_default_transmon_pair_macros,
)

# Standard OPX1000 port counts per FEM module (override at module level if needed)
MW_FEM_OUTPUTS = 8
LF_FEM_OUTPUTS = 8

# Shared readout RX inputs: every qubit on a FEM receives its readout through these.
LF_SHARED_RO_RX_I = 1
LF_SHARED_RO_RX_Q = 2
MW_SHARED_RO_RX = 1

# Slots available per OPX1000 chassis, used when synthesizing a controller inventory
# for a requested number of qubits (see build_dummy_quam_for_qubits).
FEM_SLOTS_PER_CONTROLLER = 8

# ── LF-FEM port allocation ─────────────────────────────────────────────────────
# The last two analog outputs of every LF-FEM are reserved as a *shared* readout
# TX I+Q pair.  All qubits on the same FEM drive their readout tones through this
# single output pair at different intermediate frequencies (multiplexed readout).
# The remaining outputs are split evenly among the qubits for their drive lines
# (and, when flux-tunable, one extra output per qubit for the flux/z line).
#
#   Fixed-frequency: 2 outputs/qubit (drive I+Q)       → (8−2)÷2 = 3 qubits/FEM
#   Flux-tunable:    3 outputs/qubit (drive I+Q + flux) → (8−2)÷3 = 2 qubits/FEM
#
# Port layout on an 8-output LF-FEM — fixed-frequency:
#   out 1,2 → q0 drive I,Q    out 3,4 → q1 drive I,Q    out 5,6 → q2 drive I,Q
#   out 7,8 → shared RO TX I,Q (all qubits)
#   in  1,2 → shared RO RX I,Q (all qubits)
#
# Port layout — flux-tunable:
#   out 1,2,3 → q0 drive I,Q + flux    out 4,5,6 → q1 drive I,Q + flux
#   out 7,8   → shared RO TX I,Q
#   in  1,2   → shared RO RX I,Q
#
# Note: MW-FEM has no baseband analog outputs so flux-tunable qubits are not
# supported there; the MW path is always fixed-frequency.

LF_DRIVE_PORTS_PER_QUBIT = 2           # drive I + drive Q
LF_DRIVE_AND_FLUX_PORTS_PER_QUBIT = 3  # drive I + drive Q + flux
LF_SHARED_RO_TX_PORTS = 2              # one I+Q pair per FEM, shared by all qubits
MW_PORTS_PER_QUBIT = 2                 # MW-FEM: drive (1) + RO TX (1); RO RX shared

QUBITS_PER_LF_FEM = (
    (LF_FEM_OUTPUTS - LF_SHARED_RO_TX_PORTS) // LF_DRIVE_PORTS_PER_QUBIT          # 3
)
QUBITS_PER_LF_FEM_FLUX = (
    (LF_FEM_OUTPUTS - LF_SHARED_RO_TX_PORTS) // LF_DRIVE_AND_FLUX_PORTS_PER_QUBIT  # 2
)
QUBITS_PER_MW_FEM = MW_FEM_OUTPUTS // MW_PORTS_PER_QUBIT                           # 4

# Shared RO TX ports — fixed at the top of the output numbering space
LF_SHARED_RO_TX_I = LF_FEM_OUTPUTS - LF_SHARED_RO_TX_PORTS + 1   # 7
LF_SHARED_RO_TX_Q = LF_FEM_OUTPUTS                                 # 8

# Default gate pulse parameters used when populating a dummy QUAM.
# These are placeholder values — calibration will overwrite them.
# Mirrors the convention used in standard qua-libs initialisation scripts.
DEFAULT_PULSE_AMPLITUDE = 0.1          # V
DEFAULT_PULSE_LENGTH = 40              # ns  (multiple of 4, ≥ 16)
DEFAULT_DRAG_ALPHA = 0.0               # DRAG coefficient (no correction until calibrated)
DEFAULT_DRAG_DETUNING = 0.0            # Hz

# Placeholder frequencies so that ``machine.generate_config()`` succeeds; calibration
# overwrites them.  They follow the wiring above:
#   - Drive: every qubit has its own drive line, hence its own LO (or MW upconverter),
#     placed DEFAULT_DRIVE_IF below a per-qubit drive frequency.
#   - Readout: qubits sharing a readout line (all qubits of one LF-FEM) share one LO and
#     are multiplexed at distinct IFs, DEFAULT_READOUT_IF + k * DEFAULT_READOUT_IF_SPACING
#     for the k-th qubit on the line (at most 3 per LF-FEM, so IFs stay well inside
#     ±400 MHz).  Each readout line gets its own LO, DEFAULT_READOUT_LINE_SPACING apart,
#     so every qubit's readout frequency is distinct.
DEFAULT_ANHARMONICITY = -200e6         # Hz
DEFAULT_DRIVE_RF = 5.0e9               # Hz, q0 drive frequency (f_01)
DEFAULT_DRIVE_SPACING = 20e6           # Hz between consecutive qubits' drive frequencies
DEFAULT_DRIVE_IF = 100e6               # Hz, drive LO/upconverter placed this far below f_01
DEFAULT_READOUT_LO = 7.0e9             # Hz, LO of the first readout line
DEFAULT_READOUT_LINE_SPACING = 200e6   # Hz between consecutive readout lines' LOs
DEFAULT_READOUT_IF = 50e6              # Hz, IF of the first qubit on a readout line
DEFAULT_READOUT_IF_SPACING = 20e6      # Hz between multiplexed qubits on one line

# Placeholder network used when no QMM is available (build_dummy_quam_for_qubits).
DUMMY_NETWORK = {"host": "127.0.0.1", "port": None, "cluster_name": "dummy_cluster"}


def create_dummy_quam(
    qmm: QuantumMachinesManager,
    quam_class: Type[AnyQuam],
) -> AnyQuam:
    """Alias for build_dummy_quam — kept for backwards compatibility."""
    return build_dummy_quam(qmm, quam_class)


def build_dummy_quam(
    qmm: QuantumMachinesManager,
    quam_class: Type[AnyQuam],
) -> AnyQuam:
    """Build a dummy QUAM by querying a live QuantumMachinesManager.

    Calls qmm.get_devices() to discover hardware, then sweeps all controllers
    and their FEMs in slot order, allocating qubits greedily:

    - LF-FEM (fixed-frequency): 3 qubits per FEM.  Each qubit gets 2 dedicated
      drive outputs (I+Q); all qubits share the last 2 outputs as the resonator TX
      pair (multiplexed readout) and the 2 analog inputs as the resonator RX pair.
      Frequency converters are stub FrequencyConverter(LocalOscillator(frequency=None))
      objects representing an external LO — no Octave is referenced.
    - LF-FEM (flux-tunable): 2 qubits per FEM.  Each qubit gets 3 dedicated outputs
      (drive I+Q + flux); the last 2 outputs and 2 inputs are again shared for RO.
    - MW-FEM: 4 qubits per FEM (2 MW outputs each, 1 shared input).
      MW-FEM has no baseband analog outputs, so flux tunability is not supported.

    Qubits are named q0, q1, q2, ... globally across all FEMs.
    Nearest-neighbour qubit pairs are added as bare QubitPair objects (no gate channels).
    machine.network is populated with host, port, and cluster_name from the QMM connection.

    Args:
        qmm: Connected QuantumMachinesManager instance.
        quam_class: QuAM root class to instantiate (e.g. FixedFrequencyQuam).

    Returns:
        Populated QuAM instance — not saved to disk; caller decides where to persist it.

    Raises:
        ValueError: If no controllers or FEMs are found.
    """
    controllers = qmm.get_devices().controllers
    network = _extract_network(qmm)
    return _build_from_controllers(controllers, network, quam_class)


def build_dummy_quam_for_qubits(
    num_qubits: int,
    quam_class: Type[AnyQuam],
    fem_type: str = "LF",
    network: Mapping[str, Any] | None = None,
) -> AnyQuam:
    """Build a dummy QUAM with exactly ``num_qubits`` qubits, without any QMM.

    Synthesizes the smallest OPX1000 inventory (controllers ``con1``, ``con2``, ...,
    each with up to :data:`FEM_SLOTS_PER_CONTROLLER` FEMs of ``fem_type``) that holds
    ``num_qubits`` qubits under the same allocation rules as :func:`build_dummy_quam`,
    then builds the machine from it, keeping only the first ``num_qubits`` qubits.

    Args:
        num_qubits: Number of qubits (``q0`` ... ``q{num_qubits - 1}``).
        quam_class: QuAM root class to instantiate (e.g. FluxTunableQuam).
        fem_type: ``"LF"`` or ``"MW"`` for every synthesized FEM. ``"MW"`` is
            fixed-frequency only.
        network: Network dict for ``machine.network``; defaults to :data:`DUMMY_NETWORK`.

    Raises:
        ValueError: If ``num_qubits`` < 1, ``fem_type`` is unknown, or ``"MW"`` is
            requested for a flux-tunable QuAM class.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")
    if fem_type not in ("LF", "MW"):
        raise ValueError(f"fem_type must be 'LF' or 'MW', got {fem_type!r}")
    flux_tunable = _is_flux_tunable(quam_class)
    if fem_type == "MW" and flux_tunable:
        raise ValueError("MW-FEMs have no baseband outputs, so they cannot host flux-tunable qubits")

    qubits_per_fem = QUBITS_PER_MW_FEM if fem_type == "MW" else _qubits_per_lf_fem(flux_tunable)
    num_fems = -(-num_qubits // qubits_per_fem)
    controllers = {}
    for fem_index in range(num_fems):
        controller = f"con{fem_index // FEM_SLOTS_PER_CONTROLLER + 1}"
        slot = fem_index % FEM_SLOTS_PER_CONTROLLER + 1
        controllers.setdefault(controller, {})[slot] = fem_type
    return _build_from_controllers(
        controllers, dict(network or DUMMY_NETWORK), quam_class, max_qubits=num_qubits
    )


def _is_flux_tunable(quam_class: Type[AnyQuam]) -> bool:
    """Whether the requested QuAM class uses flux-tunable qubits."""
    qubit_type = getattr(quam_class, "qubit_type", None)
    return isinstance(qubit_type, type) and issubclass(qubit_type, FluxTunableTransmon)


def _qubits_per_lf_fem(flux_tunable: bool) -> int:
    return QUBITS_PER_LF_FEM_FLUX if flux_tunable else QUBITS_PER_LF_FEM


def _build_from_controllers(
    controllers: dict[str, Any],
    network: dict,
    quam_class: Type[AnyQuam],
    max_qubits: int | None = None,
) -> AnyQuam:
    """Core builder used by build_dummy_quam; also callable directly in tests
    by passing a mock controllers dict and a hand-crafted network dict.

    Args:
        controllers: Mapping returned by ``QuantumMachinesManager.get_devices().controllers``
            (controller name → controller object with a ``fems`` mapping), or a plain
            ``{controller_name: {slot: "LF" | "MW"}}`` dict.
        network: Dict with keys 'host', 'port', 'cluster_name'.
        quam_class: QuAM root class to instantiate.
        max_qubits: Stop allocating once this many qubits are wired (all available
            FEM capacity is used when ``None``).

    Raises:
        TypeError: If controllers is not a dict.
        ValueError: If no controllers or FEMs are found.
    """
    if not isinstance(controllers, dict):
        raise TypeError(
            "Expected dictionary of controllers from QuantumMachinesManager.get_devices().controllers."
        )

    # Flux-tunable QuAMs get one extra (flux) output per qubit, hence fewer qubits per LF-FEM.
    is_flux_tunable = _is_flux_tunable(quam_class)
    qubits_per_lf_fem = _qubits_per_lf_fem(is_flux_tunable)

    wiring: dict = {}
    qubit_idx = 0

    def has_room() -> bool:
        return max_qubits is None or qubit_idx < max_qubits

    for ctrl_name, ctrl_info in controllers.items():
        fems = _extract_fems(ctrl_info)
        for slot in sorted(fems.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            fem_type = _extract_fem_type(fems[slot]) or "LF"
            if fem_type == "MW":
                for k in range(QUBITS_PER_MW_FEM):
                    if not has_room():
                        break
                    _add_mw_qubit_wiring(wiring, ctrl_name, slot, k, qubit_idx)
                    qubit_idx += 1
            else:
                for k in range(qubits_per_lf_fem):
                    if not has_room():
                        break
                    _add_lf_qubit_wiring(
                        wiring, ctrl_name, slot, k, qubit_idx, include_flux=is_flux_tunable
                    )
                    qubit_idx += 1

    if qubit_idx == 0:
        raise ValueError(
            "No FEMs found in any controller from QuantumMachinesManager.get_devices().controllers."
        )

    machine = quam_class()
    machine.ports = FEMPortsContainer()
    machine.wiring = wiring
    machine.network = network

    add_external_mixers(machine)
    add_ports(machine)
    add_transmons(machine)
    _add_qubit_pairs(machine)
    _add_pulses_and_macros(machine)
    _add_placeholder_frequencies(machine)

    return machine


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _extract_network(qmm: QuantumMachinesManager) -> dict:
    """Host, port and cluster name of the QMM connection, for ``machine.network``."""
    server_details = getattr(qmm, "_server_details", None)
    return {
        "host": getattr(server_details, "host", None),
        "port": getattr(server_details, "port", None),
        "cluster_name": getattr(qmm, "cluster_name", None),
    }


def _extract_fems(ctrl_info: Any) -> Mapping[Any, Any]:
    """The ``{slot: fem}`` mapping of a controller.

    Accepts an OPX1000 controller object from ``get_devices()`` (``.fems``) or a plain
    mapping; a controller without FEMs (e.g. an OPX+) yields an empty mapping.
    """
    if isinstance(ctrl_info, Mapping):
        return ctrl_info.get("fems", ctrl_info)
    return getattr(ctrl_info, "fems", None) or {}


def _extract_fem_type(fem_info: Any) -> str | None:
    """``"LF"``/``"MW"`` for a FEM entry: qm reports the type string directly."""
    if isinstance(fem_info, str):
        return fem_info.upper()
    fem_type = getattr(fem_info, "type", None) or getattr(fem_info, "fem_type", None)
    return str(fem_type).upper() if fem_type is not None else None


# ---------------------------------------------------------------------------
# Wiring builders
# ---------------------------------------------------------------------------

def _add_mw_qubit_wiring(
    wiring: dict, ctrl: str, slot: Any, k: int, global_idx: int
) -> None:
    """Append MW-FEM wiring entries for one qubit.

    k: qubit index within this FEM (0-based), determines which output ports to use.
    global_idx: qubit's global counter (determines name q{global_idx}).
    """
    qname = f"q{global_idx}"
    drive_out = 2 * k + 1
    ro_out = 2 * k + 2

    wiring.setdefault("qubits", {})[qname] = {
        WiringLineType.DRIVE.value: {
            "opx_output": f"#/ports/mw_outputs/{ctrl}/{slot}/{drive_out}",
        },
        WiringLineType.RESONATOR.value: {
            "opx_output": f"#/ports/mw_outputs/{ctrl}/{slot}/{ro_out}",
            "opx_input": f"#/ports/mw_inputs/{ctrl}/{slot}/{MW_SHARED_RO_RX}",
        },
    }


def _add_lf_qubit_wiring(
    wiring: dict,
    ctrl: str,
    slot: Any,
    k: int,
    global_idx: int,
    include_flux: bool = False,
) -> None:
    """Append LF-FEM wiring entries for one qubit.

    k          — qubit index within this FEM (0-based).
    global_idx — global qubit counter, determines name ``q{global_idx}``.
    include_flux — when True, includes a ``WiringLineType.FLUX`` entry so that
        ``add_transmons()`` creates a ``FluxLine`` on ``transmon.z``.

    Resonator TX (opx_output_I/Q) is **shared** across all qubits on the same FEM
    (multiplexed readout): every qubit on this FEM points to the same last two analog
    outputs (``LF_SHARED_RO_TX_I`` / ``LF_SHARED_RO_TX_Q``).  The OPX sums their
    readout tones, each at a different intermediate frequency.  Resonator RX inputs
    are likewise shared (ports 1 and 2).

    Only the drive output pair — and, when ``include_flux=True``, the flux output — are
    qubit-specific.  Frequency converters are stub external-mixer references so that
    ``add_external_mixers()`` creates ``FrequencyConverter(LocalOscillator(frequency=None))``
    placeholders without referencing an Octave.

    Port layout on an 8-output LF-FEM:
        Fixed-frequency (include_flux=False):
            drive_I = 2k+1,  drive_Q = 2k+2   (k = 0,1,2)
            shared RO TX I = 7,  Q = 8
        Flux-tunable (include_flux=True):
            drive_I = 3k+1,  drive_Q = 3k+2,  flux = 3k+3   (k = 0,1)
            shared RO TX I = 7,  Q = 8
    """
    qname = f"q{global_idx}"
    ports_per_qubit = (
        LF_DRIVE_AND_FLUX_PORTS_PER_QUBIT if include_flux else LF_DRIVE_PORTS_PER_QUBIT
    )
    drive_I = ports_per_qubit * k + 1
    drive_Q = ports_per_qubit * k + 2

    qubit_wiring: dict = {
        WiringLineType.DRIVE.value: {
            "opx_output_I": f"#/ports/analog_outputs/{ctrl}/{slot}/{drive_I}",
            "opx_output_Q": f"#/ports/analog_outputs/{ctrl}/{slot}/{drive_Q}",
            "frequency_converter_up": f"#/mixers/mixer_{qname}_drive",
        },
        WiringLineType.RESONATOR.value: {
            # Shared TX: all qubits on this FEM point to the same last two outputs
            "opx_output_I": f"#/ports/analog_outputs/{ctrl}/{slot}/{LF_SHARED_RO_TX_I}",
            "opx_output_Q": f"#/ports/analog_outputs/{ctrl}/{slot}/{LF_SHARED_RO_TX_Q}",
            # Shared RX (was already shared)
            "opx_input_I": f"#/ports/analog_inputs/{ctrl}/{slot}/{LF_SHARED_RO_RX_I}",
            "opx_input_Q": f"#/ports/analog_inputs/{ctrl}/{slot}/{LF_SHARED_RO_RX_Q}",
            "frequency_converter_up": f"#/mixers/mixer_{qname}_resonator",
            "frequency_converter_down": f"#/mixers/mixer_{qname}_resonator",
        },
    }

    if include_flux:
        flux_out = ports_per_qubit * k + 3
        qubit_wiring[WiringLineType.FLUX.value] = {
            "opx_output": f"#/ports/analog_outputs/{ctrl}/{slot}/{flux_out}",
        }

    wiring.setdefault("qubits", {})[qname] = qubit_wiring


# ---------------------------------------------------------------------------
# Qubit pairs, pulses and macros
# ---------------------------------------------------------------------------

def _add_qubit_pairs(machine: AnyQuam) -> None:
    """Add a bare qubit pair (no coupler, cross-resonance or ZZ channel) per neighbouring
    qubits ``q{i}``/``q{i+1}``, following quam-builder's ``q{i}-{i+1}`` pair id convention."""
    qubit_names = list(machine.qubits)
    machine.active_qubit_pair_names = []
    for control, target in zip(qubit_names, qubit_names[1:]):
        pair_id = f"{control}-{target[1:]}"
        qubit_pair = machine.qubit_pair_type(
            id=pair_id,
            qubit_control=f"#/qubits/{control}",
            qubit_target=f"#/qubits/{target}",
        )
        machine.qubit_pairs[qubit_pair.name] = qubit_pair
        machine.active_qubit_pair_names.append(qubit_pair.name)


def _add_pulses_and_macros(machine: AnyQuam) -> None:
    """Default pulses, a DragCosine single-qubit gate set (aliased to ``x180``,
    ``x90``, ``-x90``, ``y180``, ``y90``, ``-y90``) and default macros on every qubit,
    then default pulses and macros (CZ for flux-tunable pairs) on every pair."""
    for transmon in machine.qubits.values():
        add_default_transmon_pulses(transmon)
        add_DragCosine_pulses(
            transmon,
            amplitude=DEFAULT_PULSE_AMPLITUDE,
            length=DEFAULT_PULSE_LENGTH,
            alpha=DEFAULT_DRAG_ALPHA,
            detuning=DEFAULT_DRAG_DETUNING,
        )
        add_default_transmon_macros(transmon)
    for qubit_pair in machine.qubit_pairs.values():
        add_default_transmon_pair_pulses(qubit_pair)
        add_default_transmon_pair_macros(qubit_pair)


def _set_carrier(channel: Any, rf_frequency: float, lo_frequency: float) -> None:
    """Set ``channel``'s RF frequency and its LO: the stub external mixer's local
    oscillator on an IQ (LF-FEM) channel, the port upconverter on an MW-FEM channel."""
    channel.RF_frequency = rf_frequency
    converter = getattr(channel, "frequency_converter_up", None)
    if converter is not None and getattr(converter, "local_oscillator", None) is not None:
        converter.local_oscillator.frequency = lo_frequency
    port = getattr(channel, "opx_output", None)
    if port is not None and hasattr(port, "upconverter_frequency"):
        port.upconverter_frequency = lo_frequency


def _readout_line(machine: AnyQuam, qubit_name: str) -> str:
    """The readout TX port reference a qubit transmits on; qubits sharing it are multiplexed."""
    resonator_wiring = machine.wiring["qubits"][qubit_name][WiringLineType.RESONATOR.value]
    key = "opx_output_I" if "opx_output_I" in resonator_wiring else "opx_output"
    return resonator_wiring.get_unreferenced_value(key)


def _add_placeholder_frequencies(machine: AnyQuam) -> None:
    """Fill the calibration values that config generation needs (anharmonicity for the
    DRAG waveforms, qubit and resonator frequencies, LO/upconverter frequencies) with
    placeholders consistent with the wiring (see the ``DEFAULT_*`` frequency constants)."""
    readout_lines: dict[str, int] = {}
    qubits_on_line: dict[str, int] = {}
    for index, transmon in enumerate(machine.qubits.values()):
        drive_rf = DEFAULT_DRIVE_RF + index * DEFAULT_DRIVE_SPACING
        transmon.anharmonicity = DEFAULT_ANHARMONICITY
        transmon.f_01 = drive_rf
        if transmon.xy is not None:
            _set_carrier(transmon.xy, drive_rf, drive_rf - DEFAULT_DRIVE_IF)
        if transmon.resonator is not None:
            line = _readout_line(machine, transmon.name)
            line_index = readout_lines.setdefault(line, len(readout_lines))
            position = qubits_on_line.get(line, 0)
            qubits_on_line[line] = position + 1
            readout_lo = DEFAULT_READOUT_LO + line_index * DEFAULT_READOUT_LINE_SPACING
            readout_if = DEFAULT_READOUT_IF + position * DEFAULT_READOUT_IF_SPACING
            _set_carrier(transmon.resonator, readout_lo + readout_if, readout_lo)
