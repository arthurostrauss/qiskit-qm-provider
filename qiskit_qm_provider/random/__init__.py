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

"""Python counterpart of QUA's ``random`` library.

:class:`Random` reproduces the draws of a :class:`qm.qua.Random` on the host, and
:func:`box_muller_pair` / :func:`rand_gauss_box_muller` provide a matching pair of
Gaussian samplers. Replication requires the number and order of draws to be fixed
ahead of time; see the "Random numbers" guide.

This subpackage is intentionally not re-exported from :mod:`qiskit_qm_provider`, so
it does not clash with ``qm.qua.Random`` under star-imports::

    from qiskit_qm_provider.random import Random
"""

from .gaussian import BoxMullerTables, box_muller_pair, rand_gauss_box_muller
from .lcg import Random

__all__ = [
    "Random",
    "BoxMullerTables",
    "box_muller_pair",
    "rand_gauss_box_muller",
]
