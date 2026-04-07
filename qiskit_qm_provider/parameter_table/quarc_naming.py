# Copyright 2026 Arthur Strauss
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deterministic Quarc **struct** naming aligned with OPNIC ``ParameterTable`` / ``Parameter`` pool ids."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .parameter import Parameter
    from .parameter_table import ParameterTable


def default_quarc_struct_name(obj: Union["ParameterTable", "Parameter"]) -> str:
    """
    Deterministic struct key used in Quarc codegen and in ``module.json`` ``_structs``.

    - :class:`~qiskit_qm_provider.parameter_table.ParameterTable`: ``Packet_{table.name}_{table._id}``
    - Standalone :class:`~qiskit_qm_provider.parameter_table.Parameter` (OPNIC, not in any table):
      ``Packet_{parameter.name}_{stream_id}`` where ``stream_id`` is allocated from
      :meth:`~qiskit_qm_provider.parameter_table.ParameterPool.get_id` on first use.

    **OPX↔OPNIC edge stream ids** (incoming/outgoing) are attached by
    :meth:`~qiskit_qm_provider.parameter_table.ParameterPool.prepare_opnic_quarc_hybrid_packets`.
    """
    from .parameter import Parameter
    from .parameter_table import ParameterTable

    if isinstance(obj, ParameterTable):
        return f"Packet_{obj.name}_{obj._id}"
    if isinstance(obj, Parameter):
        if obj.tables:
            raise ValueError(
                "Use the owning ParameterTable for Quarc struct naming; this Parameter belongs to a table."
            )
        return f"Packet_{obj.name}_{obj.stream_id}"
    raise TypeError(f"Expected ParameterTable or Parameter, got {type(obj)!r}")


def quarc_packet_sort_key(obj: Union["ParameterTable", "Parameter"]) -> int:
    """Sort key for stable struct registration order (same counter namespace as :class:`ParameterPool`)."""
    from .parameter import Parameter
    from .parameter_table import ParameterTable

    if isinstance(obj, ParameterTable):
        return int(obj._id)
    if isinstance(obj, Parameter):
        return int(obj.stream_id)
    raise TypeError(f"Expected ParameterTable or Parameter, got {type(obj)!r}")
