# Copyright 2026 Arthur Strauss
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Protocol for objects that participate in OPNIC / Quarc packet codegen (``Parameter``, ``ParameterTable``)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .input_type import Direction


@runtime_checkable
class OpnicPacketBinding(Protocol):
    """Logical OPNIC packet target with a stable pool ``name`` and stream ``direction``."""

    @property
    def name(self) -> str: ...

    @property
    def direction(self) -> Direction: ...
