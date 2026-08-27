"""Qiskit Pulse is optional: no import-time warning, no eager Pulse imports."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

from qiskit_qm_provider.backend.qm_backend import (
    QISKIT_PULSE_AVAILABLE,
    requires_qiskit_pulse,
)


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_qiskit_pulse_available_matches_find_spec():
    assert QISKIT_PULSE_AVAILABLE is (importlib.util.find_spec("qiskit.pulse") is not None)


def test_requires_qiskit_pulse_raises_when_unavailable():
    @requires_qiskit_pulse
    def gated():
        return "ok"

    if QISKIT_PULSE_AVAILABLE:
        assert gated() == "ok"
    else:
        with pytest.raises(ImportError, match="Qiskit Pulse"):
            gated()


def test_package_import_does_not_warn_about_qiskit_pulse():
    result = _run_isolated("""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import qiskit_qm_provider  # noqa: F401

        pulse_warnings = [
            str(w.message)
            for w in caught
            if "pulse" in str(w.message).lower() or "Pulse" in str(w.message)
        ]
        if pulse_warnings:
            raise SystemExit(f"unexpected Pulse warnings: {pulse_warnings!r}")
        """)
    assert result.returncode == 0, result.stderr + result.stdout


def test_package_import_does_not_load_qiskit_pulse():
    result = _run_isolated("""
        import builtins
        import sys

        import qiskit  # noqa: F401  — baseline; Pulse must not be added by this package

        before = {m for m in sys.modules if m == "qiskit.pulse" or m.startswith("qiskit.pulse.")}
        pulse_loaded = []
        real_import = builtins.__import__

        def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "qiskit.pulse" or name.startswith("qiskit.pulse."):
                pulse_loaded.append(name)
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = tracking_import
        import qiskit_qm_provider as qmp  # noqa: F401
        from qiskit_qm_provider.backend.qm_backend import QMBackend, QISKIT_PULSE_AVAILABLE  # noqa: F401

        after = {m for m in sys.modules if m == "qiskit.pulse" or m.startswith("qiskit.pulse.")}
        new_modules = sorted(after - before)
        if pulse_loaded or new_modules:
            raise SystemExit(
                f"qiskit.pulse imported during package/backend load: "
                f"imports={pulse_loaded!r} modules={new_modules!r}"
            )
        print("QISKIT_PULSE_AVAILABLE", QISKIT_PULSE_AVAILABLE)
        """)
    assert result.returncode == 0, result.stderr + result.stdout


def test_quam_qiskit_pulse_symbols_are_lazy():
    result = _run_isolated("""
        import importlib.util
        import qiskit_qm_provider as qmp

        pulse_available = importlib.util.find_spec("qiskit.pulse") is not None
        if pulse_available != qmp.QISKIT_PULSE_AVAILABLE:
            raise SystemExit("QISKIT_PULSE_AVAILABLE mismatch")

        if pulse_available:
            from qiskit_qm_provider import QuAMQiskitPulse, FluxChannel  # noqa: F401
        else:
            try:
                from qiskit_qm_provider import QuAMQiskitPulse  # noqa: F401
            except ImportError as exc:
                if "Qiskit Pulse" not in str(exc):
                    raise SystemExit(f"unexpected ImportError: {exc}")
            else:
                raise SystemExit("expected ImportError for QuAMQiskitPulse without Pulse")
        """)
    assert result.returncode == 0, result.stderr + result.stdout
