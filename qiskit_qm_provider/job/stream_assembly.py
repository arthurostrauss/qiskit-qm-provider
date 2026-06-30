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

"""Shared helpers for assembling QUA stream data into Qiskit primitive containers.

Author: Arthur Strauss
Date: 2026-06-30
"""

from __future__ import annotations

import numpy as np

from qiskit.primitives.containers import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub


def bit_array_from_stream(raw, bit_width: int, target_shape: tuple[int, ...]) -> BitArray:
    """Assemble a classified :class:`~qiskit.primitives.containers.BitArray` from a QUA stream."""
    data = np.asarray(raw, dtype=int).reshape(target_shape)
    return BitArray.from_samples(data.reshape(-1).tolist(), bit_width).reshape(target_shape)


def bit_array_from_measurement_stream(pub: SamplerPub, raw, bit_width: int) -> BitArray:
    """Assemble a sampler :class:`~qiskit.primitives.containers.BitArray` from a QUA stream."""
    return bit_array_from_stream(raw, bit_width, pub.shape + (pub.shots,))
