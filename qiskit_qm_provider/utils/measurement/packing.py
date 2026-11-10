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

"""Pack compiler-wired classical bits into a QUA integer.

Author: Arthur Strauss
Date: 2026-10-29
"""

from qm.qua import Cast


def _measurement_var_is_array(var) -> bool:
    """Return whether a compiler-wired classical output is a QUA bool array.

    Scalar outputs are bool :class:`~qm.qua._expressions.QuaVariable` instances;
    multi-bit (and some size-``1``) outputs are
    :class:`~qm.qua._expressions.QuaArrayVariable` (or subclasses).
    """
    from qm.qua._expressions import QuaArrayVariable

    return isinstance(var, QuaArrayVariable)


def pack_register_to_int(var, size: int):
    """Pack classical bits into a single integer (LSB = bit index 0).

    Packing follows the wired QUA variable shape: arrays are indexed as ``var[i]``;
    scalars are cast directly (only valid when ``size == 1``).

    Args:
        var: Compiler-owned QUA bool scalar or bool array from ``result_program``.
        size: Number of bits to pack from ``var``.

    Raises:
        ValueError: If ``size`` is not positive, or a multi-bit output is wired to a
            scalar QUA variable.
    """
    if size < 1:
        raise ValueError(f"pack_register_to_int requires size >= 1, got {size}.")

    if _measurement_var_is_array(var):
        return sum(
            (((1 << i) * Cast.to_int(var[i])) for i in range(1, size)),
            start=Cast.to_int(var[0]),
        )

    if size != 1:
        raise ValueError(
            f"Expected a QUA bool array for a {size}-bit classical register, " f"got scalar {type(var).__name__}."
        )
    return Cast.to_int(var)
