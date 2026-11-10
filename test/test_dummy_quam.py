"""The dummy QuAM builder in :mod:`test.dummy_quam` produces complete, config-valid machines."""

from types import SimpleNamespace

import pytest

pytest.importorskip("quam_builder.builder.superconducting.modify_quam")

from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam  # noqa: E402

from . import dummy_quam  # noqa: E402
from .dummy_quam import (  # noqa: E402
    DUMMY_NETWORK,
    FEM_SLOTS_PER_CONTROLLER,
    QUBITS_PER_LF_FEM_FLUX,
    build_dummy_quam,
    build_dummy_quam_for_qubits,
)


def _port(machine, qubit: str, line: str, key: str) -> str:
    """The raw ``#/ports/...`` reference of a wiring entry (wiring values resolve to ports)."""
    return machine.wiring["qubits"][qubit][line].get_unreferenced_value(key)


@pytest.fixture(autouse=True)
def _writable_quam_state(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAM_STATE_PATH", str(tmp_path))


@pytest.mark.parametrize(
    "quam_class, fem_type, num_qubits",
    [
        (FluxTunableQuam, "LF", 1),
        (FluxTunableQuam, "LF", 2),
        (FluxTunableQuam, "LF", 5),
        (FluxTunableQuam, "LF", 17),
        (FixedFrequencyQuam, "LF", 7),
        (FixedFrequencyQuam, "MW", 6),
    ],
)
def test_builds_a_complete_config_valid_machine(quam_class, fem_type, num_qubits):
    machine = build_dummy_quam_for_qubits(num_qubits, quam_class, fem_type)

    assert [qubit.name for qubit in machine.active_qubits] == [f"q{index}" for index in range(num_qubits)]
    assert len(machine.active_qubit_pairs) == num_qubits - 1
    assert machine.network == DUMMY_NETWORK
    for qubit in machine.active_qubits:
        assert {"x", "y", "sx", "sy", "rz", "measure", "reset", "delay", "id"} <= set(qubit.macros)
        # The DragCosine gate set is aliased to the canonical names the provider uses.
        assert qubit.get_pulse("x180").length == dummy_quam.DEFAULT_PULSE_LENGTH
        assert qubit.anharmonicity == dummy_quam.DEFAULT_ANHARMONICITY
        assert (getattr(qubit, "z", None) is not None) == (quam_class is FluxTunableQuam)
    if quam_class is FluxTunableQuam:
        assert all("cz" in pair.macros for pair in machine.active_qubit_pairs)

    config = machine.generate_config()

    channels_per_qubit = 3 if quam_class is FluxTunableQuam else 2
    assert len(config["elements"]) == channels_per_qubit * num_qubits


def test_inventory_spills_onto_further_controllers():
    num_qubits = QUBITS_PER_LF_FEM_FLUX * FEM_SLOTS_PER_CONTROLLER + 1  # one FEM past con1
    machine = build_dummy_quam_for_qubits(num_qubits, FluxTunableQuam)

    assert "/con2/1/" in _port(machine, f"q{num_qubits - 1}", "z", "opx_output")
    assert "/con1/8/" in _port(machine, f"q{num_qubits - 2}", "z", "opx_output")


def test_qubits_on_one_fem_share_the_readout_lines():
    machine = build_dummy_quam_for_qubits(2, FluxTunableQuam)

    assert _port(machine, "q0", "rr", "opx_output_I") == _port(machine, "q1", "rr", "opx_output_I")
    assert _port(machine, "q0", "rr", "opx_input_I") == _port(machine, "q1", "rr", "opx_input_I")
    assert _port(machine, "q0", "xy", "opx_output_I") != _port(machine, "q1", "xy", "opx_output_I")


def test_readout_is_multiplexed_per_shared_line():
    """Qubits on one LF-FEM share the readout LO and sit at distinct IFs; separate FEMs get
    separate LOs, so every readout frequency is distinct. Drive LOs are per qubit."""
    machine = build_dummy_quam_for_qubits(5, FixedFrequencyQuam)  # FEM 1: q0-q2, FEM 2: q3-q4
    resonators = [qubit.resonator for qubit in machine.active_qubits]
    los = [resonator.frequency_converter_up.local_oscillator.frequency for resonator in resonators]
    ifs = [resonator.RF_frequency - lo for resonator, lo in zip(resonators, los)]

    assert los[0] == los[1] == los[2] != los[3] == los[4]
    assert len(set(ifs[:3])) == 3 and len(set(ifs[3:])) == 2
    assert all(abs(intermediate) < 400e6 for intermediate in ifs)
    assert len({resonator.RF_frequency for resonator in resonators}) == 5
    drive_los = [qubit.xy.frequency_converter_up.local_oscillator.frequency for qubit in machine.active_qubits]
    assert len(set(drive_los)) == 5
    machine.generate_config()


def test_build_from_a_qmm_device_inventory():
    """build_dummy_quam reads controllers (OPX1000 objects with a ``fems`` mapping) and
    network details from the QMM."""
    qmm = SimpleNamespace(
        get_devices=lambda: SimpleNamespace(controllers={"con1": SimpleNamespace(fems={2: "MW", 1: "LF"})}),
        _server_details=SimpleNamespace(host="10.0.0.1", port=9510),
        cluster_name="my_cluster",
    )

    machine = build_dummy_quam(qmm, FixedFrequencyQuam)

    # Slot 1 (LF, 3 fixed-frequency qubits) is allocated before slot 2 (MW, 4 qubits).
    assert len(machine.active_qubits) == dummy_quam.QUBITS_PER_LF_FEM + dummy_quam.QUBITS_PER_MW_FEM
    assert "analog_outputs/con1/1/" in _port(machine, "q0", "xy", "opx_output_I")
    assert "mw_outputs/con1/2/" in _port(machine, "q3", "xy", "opx_output")
    assert machine.network == {"host": "10.0.0.1", "port": 9510, "cluster_name": "my_cluster"}
    machine.generate_config()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"num_qubits": 0, "quam_class": FluxTunableQuam}, "at least 1"),
        ({"num_qubits": 2, "quam_class": FluxTunableQuam, "fem_type": "MW"}, "cannot host flux-tunable"),
        ({"num_qubits": 2, "quam_class": FixedFrequencyQuam, "fem_type": "XX"}, "'LF' or 'MW'"),
    ],
)
def test_rejects_invalid_requests(kwargs, error):
    with pytest.raises(ValueError, match=error):
        build_dummy_quam_for_qubits(**kwargs)


def test_rejects_a_non_dict_controller_inventory():
    with pytest.raises(TypeError, match="Expected dictionary"):
        dummy_quam._build_from_controllers([], {}, FluxTunableQuam)
