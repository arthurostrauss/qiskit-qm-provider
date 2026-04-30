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

"""Convert :class:`ParameterTable` field info into Quarc :func:`quarc.Struct` types.

These helpers do **not** create :class:`StructSpec` instances or
:class:`QuaStructHandle` objects — that step (and the stream-id consumption that comes
with it) lives on :meth:`ParameterTable._emit_to_module` and is performed only when
:meth:`ParameterPool.to_quarc_module` (or :meth:`ParameterPool.from_quarc_module`)
finally hands the table a Quarc :class:`BaseModule` to register on. This lets us defer
the irreversible / global-state-touching part of the conversion until the user actually
asks for the module to be assembled.

Public helpers:

* :func:`quarc_atomic_for` — qiskit-qm ``qua_type`` → Quarc atomic (``int|float|bool``).
* :func:`quarc_annotation_for` — :class:`Parameter` → ``Scalar[T] | Array[T, N]``.
* :func:`quarc_direction_for` — qiskit-qm :class:`Direction` → ``quarc.Direction``.
* :func:`build_quarc_struct` — build the ``quarc.Struct`` *type* (a Python class with
  the right annotations) for a :class:`ParameterTable` or single :class:`Parameter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from qm.qua import fixed

from .input_type import Direction

if TYPE_CHECKING:
    from .parameter import Parameter
    from .parameter_table import ParameterTable


_QUA_TO_QUARC_ATOMIC: Dict[Any, type] = {int: int, bool: bool, fixed: float}


def quarc_atomic_for(qua_type: Any) -> type:
    """Map a qiskit-qm-provider ``qua_type`` (``int`` / ``bool`` / ``fixed``) to a
    Quarc atomic type (``int`` / ``bool`` / ``float``)."""
    try:
        return _QUA_TO_QUARC_ATOMIC[qua_type]
    except KeyError as exc:
        raise TypeError(
            f"Cannot map QUA type {qua_type!r} to a Quarc atomic; expected int, bool, or fixed."
        ) from exc


def quarc_annotation_for(parameter: "Parameter") -> Any:
    """Return the ``Scalar[T]`` / ``Array[T, N]`` annotation that mirrors the parameter
    shape (scalar vs. 1D array of length ``parameter.length``)."""
    from quarc import Array, Scalar

    atomic = quarc_atomic_for(parameter.type)
    if parameter.is_array:
        return Array[atomic, parameter.length]
    return Scalar[atomic]


def quarc_direction_for(direction: Direction) -> Any:
    """Translate a qiskit-qm-provider :class:`Direction` to its Quarc counterpart.

    qiskit-qm ``Direction`` is described from the OPX's perspective:

    * ``OUTGOING`` (OPNIC -> OPX) maps to Quarc ``INCOMING`` (data flows into QUA).
    * ``INCOMING`` (OPX -> OPNIC) maps to Quarc ``OUTGOING`` (data flows out of QUA).
    * ``BOTH`` is bidirectional in both vocabularies.
    """
    from quarc import Direction as QuarcDirection

    if direction == Direction.OUTGOING:
        return QuarcDirection.INCOMING
    if direction == Direction.INCOMING:
        return QuarcDirection.OUTGOING
    if direction == Direction.BOTH:
        return QuarcDirection.BOTH
    raise ValueError(f"Cannot map qiskit-qm direction {direction!r} to a Quarc direction.")


def build_quarc_struct(
    source: "ParameterTable | Parameter",
    *,
    struct_name: Optional[str] = None,
) -> type:
    """Build a Quarc ``Struct`` *type* mirroring a ``ParameterTable`` (multi-field) or
    a single ``Parameter`` (single-field).

    Returns the Python class produced by :func:`quarc.Struct`. **No stream ids are
    consumed and no Quarc :class:`BaseModule` is required for this call** — it just
    constructs the typed struct definition. The actual ``add_struct`` step is performed
    later by :meth:`ParameterTable._emit_to_module`.
    """
    from quarc import Struct

    from .parameter import Parameter
    from .parameter_table import ParameterTable

    if isinstance(source, ParameterTable):
        params: Sequence["Parameter"] = source.parameters
        default_name = source.name
    elif isinstance(source, Parameter):
        params = [source]
        default_name = source.name
    else:
        raise TypeError(
            f"build_quarc_struct expects a ParameterTable or Parameter, got "
            f"{type(source).__name__}."
        )

    name = struct_name or default_name
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"Cannot build Quarc struct: derived struct name {name!r} is not a valid "
            "Python identifier. Provide a valid name via the `name` argument of "
            "ParameterTable / Parameter."
        )

    fields = {p.name: quarc_annotation_for(p) for p in params}
    return Struct(struct_name=name, **fields)
