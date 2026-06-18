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

"""QUA program scope detection and guards for variable accessors."""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

from qm.qua._scope_management.scopes_manager import (
    NoScopeFoundException,
    scopes_manager,
)

F = TypeVar("F", bound=Callable)


def is_inside_scope() -> bool:
    """Return ``True`` when called inside an active ``with program():`` block."""
    try:
        scopes_manager.current_scope
        return True
    except NoScopeFoundException:
        return False


def current_scope_token() -> object | None:
    """Return a stable token identifying the active QUA *program* scope, or ``None``.

    Uses the program-root scope (``scopes_manager.program_scope``), which keeps a single
    identity for the whole ``with program():`` block — unchanged across nested ``for_`` /
    ``if_`` scopes. A fresh ``with program():`` yields a new object. Used to (a) bind QUA
    declaration validity to the program it was made in (``Parameter.is_declared``), and
    (b) enforce the single-program guard for OPNIC struct emission.
    """
    try:
        return scopes_manager.program_scope
    except NoScopeFoundException:
        return None


def require_qua_program(context: str) -> None:
    """Raise if not inside an active QUA program scope.

    Used inside property getters and dunder accessors where a decorator cannot run
    domain-specific validation first (e.g. OPNIC table-managed checks on
    :meth:`~.parameter_table.Parameter.declare`).

    Args:
        context: Label for the accessor (appears in the :class:`RuntimeError` message).
    """
    if not is_inside_scope():
        raise RuntimeError(f"{context} must be accessed inside `with program():`")


def requires_qua_program(fn: F) -> F:
    """Decorator: require an active QUA program scope before calling ``fn``.

    Suitable for QUA-mutating methods (``declare``, ``assign``, ``stream_back``, …).
    Do not use at the top of methods that must raise other errors first (e.g. OPNIC
    table-managed guards on :meth:`~.parameter_table.Parameter.declare`).
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_inside_scope():
            raise RuntimeError(f"{fn.__qualname__} must be called inside `with program():`")
        return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
