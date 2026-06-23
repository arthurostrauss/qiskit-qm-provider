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

"""Shared helpers for Jinja-based sync-hook generation.

The generated sync hooks must run on the IQCC cloud side without importing
``qiskit_qm_provider`` or ``numpy``. This module turns rich parameter tables into
plain-Python data (``str``/``int``/``float``/``bool``) and renders the static
push logic from Jinja templates.

Author: Arthur Strauss
Date: 2026-06-18
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from qiskit_qm_provider.parameter_table import InputType, ParameterTable

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def jinja_env() -> Environment:
    """Return a Jinja environment loading templates from the ``templates`` directory."""
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _qua_type_str(param) -> str:
    """Map a parameter's QUA type to a plain string understood by the sync hook."""
    name = getattr(param.type, "__name__", None)
    if name in ("int", "fixed", "bool"):
        return name
    # Default to fixed-point (float) coercion for anything unexpected.
    return "fixed"


def serialize_table(table: Optional[ParameterTable]) -> Optional[dict]:
    """Serialize a :class:`ParameterTable` into plain-Python data for the sync hook.

    Returns ``None`` when the table is ``None`` or carries no ``input_type`` (nothing
    to push). OPNIC tables are rejected: OPNIC delivery is not supported in the
    self-contained sync hook (the primitives raise before reaching here).
    """
    if table is None or table.input_type is None:
        return None
    if table.input_type == InputType.OPNIC:
        raise NotImplementedError(
            "OPNIC input_type is not supported in sync-hook generation; "
            "use INPUT_STREAM or IO1/IO2."
        )
    return {
        "input_type": table.input_type.value,
        "params": [{"name": param.name, "qua_type": _qua_type_str(param)} for param in table.parameters],
    }


def to_py_literal(obj) -> str:
    """Render a plain-Python object as a source literal for template injection.

    The input must consist only of ``dict``/``list``/``tuple``/``str``/``int``/
    ``float``/``bool``/``None`` so that ``repr`` produces valid, dependency-free
    Python source.
    """
    return repr(obj)
