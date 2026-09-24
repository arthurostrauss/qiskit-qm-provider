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

"""Lookup-table Box-Muller Gaussian sampling, on the controller and replayed on the host.

One call draws a single ``rand_fixed`` per dimension and turns it into a pair of
Gaussian samples. The top ``log2(n_lookup)`` bits of the 28-bit uniform draw index a
``sqrt(-2 ln u)`` table and the low bits index a cosine table; the second sample uses
the cosine table shifted by a quarter period (i.e. the sine):

``z1 = mean + std * ln[u1] * cos[u2]``, ``z2 = mean + std * ln[u1] * cos[u2 + n/4]``

:func:`rand_gauss_box_muller` emits this in QUA and :func:`box_muller_pair` replays it
in Python with the same fixed-point operations and the same lookup values
(:class:`BoxMullerTables` is the single source for both).

Author: Arthur Strauss
Date: 2026-09-24
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Sequence

from ..fixed_point import FixedPoint
from ..parameter_table._scope import requires_qua_program
from .lcg import STATE_BITS, Random

if TYPE_CHECKING:
    from qm.qua import Random as QuaRandom

_MAX_LOOKUP_BITS = STATE_BITS // 2


class BoxMullerTables:
    """Lookup tables shared by the QUA macro and its Python replay.

    ``cos[k] = cos(2*pi*k/n)`` and ``ln[k] = sqrt(-2 ln((k+1)/(n+1)))`` for
    ``k = 0..n-1``. Use :meth:`get` to reuse cached instances.

    Args:
        n_lookup: Table length; a power of two between 4 and ``2**14`` so the ``ln``
            and ``cos`` indices come from disjoint bits of the 28-bit draw.

    Raises:
        ValueError: If ``n_lookup`` is not a supported power of two.
    """

    def __init__(self, n_lookup: int = 512):
        if n_lookup < 4 or n_lookup & (n_lookup - 1) or n_lookup > 1 << _MAX_LOOKUP_BITS:
            raise ValueError(
                f"n_lookup must be a power of two between 4 and {1 << _MAX_LOOKUP_BITS}, got {n_lookup}."
            )
        self.n_lookup = n_lookup
        self.u1_shift = STATE_BITS - (n_lookup.bit_length() - 1)
        self.cos_values = [math.cos(2 * math.pi * k / n_lookup) for k in range(n_lookup)]
        self.ln_values = [math.sqrt(-2 * math.log((k + 1) / (n_lookup + 1))) for k in range(n_lookup)]
        self.cos_fixed = [FixedPoint(v) for v in self.cos_values]
        self.ln_fixed = [FixedPoint(v) for v in self.ln_values]

    @classmethod
    @functools.cache
    def get(cls, n_lookup: int = 512) -> BoxMullerTables:
        """Return a cached :class:`BoxMullerTables` for ``n_lookup``."""
        return cls(n_lookup)

    @requires_qua_program
    def declare_qua(self):
        """Declare the tables as QUA ``fixed`` arrays; must be called inside ``with program():``.

        Returns:
            Tuple ``(cos_array, ln_array)`` of QUA arrays.
        """
        from qm.qua import declare, fixed

        return declare(fixed, value=self.cos_values), declare(fixed, value=self.ln_values)


def box_muller_pair(
    rng: Random,
    mean: Sequence[FixedPoint | float],
    std: Sequence[FixedPoint | float],
    tables: BoxMullerTables | None = None,
) -> tuple[list[FixedPoint], list[FixedPoint]]:
    """Replay one :func:`rand_gauss_box_muller` call on the host.

    Draws one :meth:`Random.rand_fixed` per dimension, in order, exactly as the QUA
    macro does.

    Args:
        rng: Host generator mirroring the QUA ``Random`` passed to the macro.
        mean: Per-dimension means; floats are quantized with ``FixedPoint(...)``
            (round to nearest, as QUA's ``declare(fixed, value=...)``).
        std: Per-dimension standard deviations, same length as ``mean``.
        tables: Lookup tables; defaults to ``BoxMullerTables.get()`` (512 entries).

    Returns:
        Tuple ``(z1, z2)`` of per-dimension samples as :class:`~qiskit_qm_provider.FixedPoint`.

    Raises:
        ValueError: If ``mean`` and ``std`` have different lengths.
    """
    if len(mean) != len(std):
        raise ValueError(f"mean and std must have the same length, got {len(mean)} and {len(std)}.")
    tables = tables or BoxMullerTables.get()
    mask = tables.n_lookup - 1
    quarter = tables.n_lookup // 4
    u2_mask = (1 << tables.u1_shift) - 1
    z1, z2 = [], []
    for mu, sigma in zip(mean, std):
        mu = mu if isinstance(mu, FixedPoint) else FixedPoint(mu)
        sigma = sigma if isinstance(sigma, FixedPoint) else FixedPoint(sigma)
        raw = rng.rand_fixed().to_unsafe_int()
        u1 = raw >> tables.u1_shift
        u2 = raw & u2_mask
        radius = sigma * tables.ln_fixed[u1]
        z1.append(mu + radius * tables.cos_fixed[u2 & mask])
        z2.append(mu + radius * tables.cos_fixed[(u2 + quarter) & mask])
    return z1, z2


@requires_qua_program
def rand_gauss_box_muller(qua_rng: QuaRandom, mean, std, z1, z2, tables_qua=None, n_lookup: int = 512):
    """QUA macro drawing one Gaussian pair per dimension into ``z1`` and ``z2``.

    Loops over ``mean.length()`` and draws one ``qua_rng.rand_fixed()`` per
    dimension; :func:`box_muller_pair` replays the same call on the host.

    Args:
        qua_rng: QUA ``Random``, typically from :meth:`Random.declare_qua`.
        mean: QUA ``fixed`` array of means.
        std: QUA ``fixed`` array of standard deviations, same length as ``mean``.
        z1: QUA ``fixed`` array receiving the first sample of each pair.
        z2: QUA ``fixed`` array receiving the second sample of each pair.
        tables_qua: ``(cos_array, ln_array)`` from :meth:`BoxMullerTables.declare_qua`.
            Pass them when calling the macro several times to declare the tables once;
            when ``None`` they are declared on each call.
        n_lookup: Table length; must match the tables passed in ``tables_qua``.

    Returns:
        Tuple ``(z1, z2)``.
    """
    from qm.qua import Cast, assign, declare, fixed, for_

    tables = BoxMullerTables.get(n_lookup)
    cos_array, ln_array = tables_qua if tables_qua is not None else tables.declare_qua()
    mask = tables.n_lookup - 1
    quarter = tables.n_lookup // 4
    i = declare(int)
    u1 = declare(int)
    u2 = declare(int)
    u = declare(fixed)
    with for_(i, 0, i < mean.length(), i + 1):
        assign(u, qua_rng.rand_fixed())
        assign(u1, Cast.unsafe_cast_int(u >> tables.u1_shift))
        assign(u2, Cast.unsafe_cast_int(u) & ((1 << tables.u1_shift) - 1))
        assign(z1[i], mean[i] + std[i] * ln_array[u1] * cos_array[u2 & mask])
        assign(z2[i], mean[i] + std[i] * ln_array[u1] * cos_array[(u2 + quarter) & mask])
    return z1, z2
