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

"""Descriptor for deprecated method aliases.

Usage::

    class MyClass:
        def new_name(self, ...): ...

        old_name = _DeprecatedAlias("new_name", removal="1.2")

Accessing ``instance.old_name`` emits a :class:`DeprecationWarning` pointing at the
caller's frame (``stacklevel=2``) and returns the bound canonical method, so the
call still works without any other change.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional


class _DeprecatedAlias:
    """Non-data descriptor that emits :class:`DeprecationWarning` on attribute access.

    Args:
        canonical: Name of the replacement method on the same class.
        removal: Version string when the alias will be removed (e.g. ``"1.2"``).
    """

    def __init__(self, canonical: str, removal: str) -> None:
        self._canonical = canonical
        self._removal = removal
        self._deprecated: Optional[str] = None  # filled by __set_name__

    def __set_name__(self, owner: Any, name: str) -> None:
        self._deprecated = name

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            # Class-level access (e.g. inspect.getmembers, hasattr) — return the descriptor
            # itself so introspection tools don't fire spurious DeprecationWarnings.
            return self
        warnings.warn(
            f"{self._deprecated}() is deprecated and will be removed in "
            f"v{self._removal}. Use {self._canonical}() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(obj, self._canonical)
