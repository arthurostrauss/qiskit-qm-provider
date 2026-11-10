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

"""Host-side mirror of QUA's ``Random`` linear congruential generator.

QUA's :class:`qm.qua.Random` keeps its state in a single QUA ``int`` variable and
advances it on the controller each time ``rand_int`` or ``rand_fixed`` is evaluated.
The update rule runs in firmware and is not documented by the QUA package; the
constants below were checked bit for bit against hardware:

``state <- (LCG_MULTIPLIER * state + LCG_INCREMENT) mod 2**28``

The state is advanced *before* each output. Because ``2**32`` is a multiple of
``2**28``, 32-bit wrap-around of the seed variable never changes the next state, so
only ``seed mod 2**28`` matters.

Author: Arthur Strauss
Date: 2026-09-24
"""

from __future__ import annotations

import random as _stdlib_random
from typing import TYPE_CHECKING

from ..fixed_point import FixedPoint
from ..parameter_table._scope import requires_qua_program

if TYPE_CHECKING:
    from qm.qua import Random as QuaRandom

LCG_MULTIPLIER = 137939405
"""Multiplier ``a`` of the QUA LCG."""
LCG_INCREMENT = 12345
"""Increment ``c`` of the QUA LCG."""
STATE_BITS = 28
"""Number of state bits; the modulus is ``2**STATE_BITS``."""
STATE_MASK = (1 << STATE_BITS) - 1
"""Mask reducing a value modulo ``2**STATE_BITS``."""


def _wrap_int32(value: int) -> int:
    """Wrap ``value`` to a signed 32-bit integer, as a QUA ``int`` would."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


class Random:
    """Python mirror of :class:`qm.qua.Random`.

    A :class:`Random` and a QUA ``Random`` that start from the same state produce
    the same values, provided the host calls :meth:`rand_int` / :meth:`rand_fixed`
    the same number of times, in the same order, as the QUA program does.

    Replication is only possible when the number and order of draws is fixed ahead
    of time. A draw placed under a measurement-dependent ``if_``, or inside a
    ``while_`` whose length depends on runtime data, cannot be replayed on the host.
    Seeding and reseeding are the user's responsibility on both sides: every
    ``qua_rng.set_seed(...)`` in the QUA program must be mirrored by a
    :meth:`set_seed` call with the same value at the same point of the draw sequence.

    Args:
        seed: Initial seed. When ``None``, a seed is drawn with the standard library
            ``random.randrange(2**28 - 1)``, as QUA does for an unseeded ``Random()``,
            and is available as :attr:`initial_seed` so it can be passed to QUA.
            Prefer :meth:`declare_qua` over constructing an unseeded QUA ``Random()``,
            whose compile-time seed is invisible to the host.
    """

    def __init__(self, seed: int | None = None):
        if seed is None:
            seed = _stdlib_random.randrange(STATE_MASK)
        self._initial_seed = _wrap_int32(int(seed))
        self._state = self._initial_seed
        self._draws = 0

    @property
    def initial_seed(self) -> int:
        """Seed given at construction (int32-wrapped)."""
        return self._initial_seed

    @property
    def state(self) -> int:
        """Current raw generator state, i.e. the value of QUA's seed variable."""
        return self._state

    @property
    def draws(self) -> int:
        """Number of draws since construction or the last :meth:`set_seed`."""
        return self._draws

    def set_seed(self, seed: int) -> None:
        """Mirror ``qua_rng.set_seed(seed)``: overwrite the state and reset :attr:`draws`.

        Args:
            seed: New seed, wrapped to a signed 32-bit integer like a QUA ``int``.
        """
        self._state = _wrap_int32(int(seed))
        self._draws = 0

    def _next_state(self) -> int:
        self._state = (LCG_MULTIPLIER * self._state + LCG_INCREMENT) & STATE_MASK
        self._draws += 1
        return self._state

    def rand_int(self, max_int: int) -> int:
        """Mirror ``qua_rng.rand_int(max_int)``: a pseudorandom integer in ``[0, max_int)``.

        Args:
            max_int: Exclusive upper bound, must be positive.

        Returns:
            The drawn integer.

        Raises:
            ValueError: If ``max_int`` is not positive.
        """
        max_int = int(max_int)
        if max_int <= 0:
            raise ValueError(f"max_int must be positive, got {max_int}.")
        return (self._next_state() * max_int) >> STATE_BITS

    def rand_fixed(self) -> FixedPoint:
        """Mirror ``qua_rng.rand_fixed()``: a pseudorandom 4.28 fixed in ``[0, 1)``.

        Returns:
            A :class:`~qiskit_qm_provider.FixedPoint` whose raw value is the new
            28-bit state. Use ``float(x)`` to convert it.
        """
        return FixedPoint.from_int(self._next_state())

    def rand_int_sequence(self, max_int: int, n: int) -> list[int]:
        """Draw ``n`` integers in order, as ``n`` successive :meth:`rand_int` calls."""
        return [self.rand_int(max_int) for _ in range(n)]

    def rand_fixed_sequence(self, n: int) -> list[FixedPoint]:
        """Draw ``n`` fixed values in order, as ``n`` successive :meth:`rand_fixed` calls."""
        return [self.rand_fixed() for _ in range(n)]

    def skip(self, n: int) -> None:
        """Advance the generator by ``n`` draws without producing values.

        Useful to step over draws the QUA program makes but the host does not need
        (their count must still be known). Runs in ``O(log n)`` by composing the
        affine update ``x -> a*x + c`` with itself.

        Args:
            n: Number of draws to skip, non-negative.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"Cannot skip a negative number of draws, got {n}.")
        # Accumulated map x -> acc_a*x + acc_c, and the current power x -> a_pow*x + c_pow.
        acc_a, acc_c = 1, 0
        a_pow, c_pow = LCG_MULTIPLIER, LCG_INCREMENT
        remaining = n
        while remaining:
            if remaining & 1:
                acc_a, acc_c = (acc_a * a_pow) & STATE_MASK, (acc_c * a_pow + c_pow) & STATE_MASK
            a_pow, c_pow = (a_pow * a_pow) & STATE_MASK, (c_pow * (a_pow + 1)) & STATE_MASK
            remaining >>= 1
        if n:
            self._state = (acc_a * self._state + acc_c) & STATE_MASK
        self._draws += n

    def copy(self) -> Random:
        """Return an independent generator with the same state and draw count."""
        clone = Random(self._initial_seed)
        clone._state = self._state
        clone._draws = self._draws
        return clone

    @requires_qua_program
    def declare_qua(self) -> QuaRandom:
        """Declare a QUA ``Random`` starting from this generator's current state.

        Must be called inside ``with program():``. The next draw of the returned QUA
        generator matches the next draw of this Python instance, whether or not the
        Python instance has already been used. From there on the QUA instance is
        owned by the user: reseed it with ``qua_rng.set_seed(...)`` and mirror each
        reseed with :meth:`set_seed` on the host.

        Calling this twice in one program declares two independent QUA states with
        the same seed; mirror each with its own :meth:`copy`.

        Returns:
            A :class:`qm.qua.Random` whose seed variable is initialized to
            :attr:`state`.
        """
        from qm.qua import Random as QuaRandom

        return QuaRandom(self._state)

    def __repr__(self) -> str:
        return f"Random(initial_seed={self._initial_seed}, state={self._state}, draws={self._draws})"
