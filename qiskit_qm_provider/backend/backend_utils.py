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

"""Deprecated import path for :func:`~qiskit_qm_provider.quam_macros.superconducting.add_basic_macros`.

Other helpers that used to live here now live under :mod:`qiskit_qm_provider.utils`.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

import warnings


def __getattr__(name: str):
    if name == "add_basic_macros":
        warnings.warn(
            "Importing add_basic_macros from qiskit_qm_provider.backend.backend_utils is deprecated. "
            "Import it from qiskit_qm_provider or "
            "qiskit_qm_provider.quam_macros.superconducting instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..quam_macros.superconducting import add_basic_macros

        return add_basic_macros
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        "Helpers that used to live in backend_utils are now under qiskit_qm_provider.utils."
    )
