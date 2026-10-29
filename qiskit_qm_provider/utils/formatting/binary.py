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

"""Integer to binary-string conversion.

Author: Arthur Strauss
Date: 2026-10-29
"""


def binary(val: int, num_bits: int = 0) -> str:
    """Convert an integer to a zero-padded binary string.

    Args:
        val: Integer value to convert.
        num_bits: Minimum width of the binary representation.

    Returns:
        Binary string without the ``0b`` prefix.
    """
    return bin(val)[2:].zfill(num_bits)
