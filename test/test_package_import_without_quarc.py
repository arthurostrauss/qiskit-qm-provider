"""Regression: top-level ``import qiskit_qm_provider`` must not load ``quarc``."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_top_level_import_without_quarc():
    result = _run_isolated("""
        import builtins
        import importlib.util
        import sys
        import warnings

        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, package=None):
            if name == "quarc" or name.startswith("quarc."):
                return None
            return real_find_spec(name, package)

        import importlib.util as iu
        iu.find_spec = fake_find_spec

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] == "quarc":
                raise ImportError("quarc blocked for test")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import qiskit_qm_provider as qmp

        assert "quarc" not in sys.modules
        assert qmp.QUARC_AVAILABLE is False
        try:
            _ = qmp.QiskitQMModule
        except ImportError:
            pass
        else:
            raise SystemExit("expected ImportError for QiskitQMModule")
        if not any("quarc" in str(w.message).lower() for w in caught):
            raise SystemExit("expected ImportWarning when quarc is unavailable")
        """)
    assert result.returncode == 0, result.stderr + result.stdout


def test_top_level_import_does_not_eagerly_load_quarc_when_installed():
    result = _run_isolated("""
        import builtins
        import importlib.util
        import sys

        if importlib.util.find_spec("quarc") is None:
            raise SystemExit(0)  # skip when quarc not installed

        quarc_loaded = []
        real_import = builtins.__import__

        def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "quarc" or name.startswith("quarc."):
                quarc_loaded.append(name)
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = tracking_import

        import qiskit_qm_provider as qmp

        if quarc_loaded:
            raise SystemExit(f"quarc imported during package load: {quarc_loaded!r}")
        if not qmp.QUARC_AVAILABLE:
            raise SystemExit("expected QUARC_AVAILABLE True")

        from qiskit_qm_provider import QiskitQMModule  # noqa: F401

        if not quarc_loaded:
            raise SystemExit("QiskitQMModule access should load quarc")
        """)
    if result.returncode == 0 and not result.stdout and not result.stderr:
        return  # passed or skipped (exit 0 with no quarc)
    assert result.returncode == 0, result.stderr + result.stdout
