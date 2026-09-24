# Random numbers

`qiskit_qm_provider.random` is the Python counterpart of QUA's `random` library
([`qm.qua.Random`](https://docs.quantum-machines.co/latest/)). Its
[`Random`](apidocs/stubs/qiskit_qm_provider.random.Random.rst) class reproduces, on the
host, the values a QUA `Random` draws on the controller. This lets a classical program
know exactly which random numbers the QUA program used without streaming them back.

For full signatures, see the [Random API reference](apidocs/qm_random.rst).

```python
from qiskit_qm_provider.random import Random
```

The subpackage is not re-exported from `qiskit_qm_provider`, so it does not clash with
`qm.qua.Random` under `from qm.qua import *`.

## The replication contract

A Python `Random` and a QUA `Random` that start from the same state produce the same
values **if the host calls the Python generator the same number of times, in the same
order, as the QUA program does**. Nothing more is guaranteed, and nothing more is
needed:

- **Seeding and reseeding are your responsibility on both sides.** Every
  `qua_rng.set_seed(expr)` in the QUA program must be mirrored by
  `py_rng.set_seed(value)` with the same value, at the same point of the draw
  sequence. The provider has no notion of iterations or loops.
- **The number and order of draws must be known ahead of time.** A fixed-count `for_`
  loop, or a loop over a length the host also knows, is fine. A draw that depends on
  runtime data cannot be replayed.

```python
# Replayable: the host knows there are exactly n_avg draws.
with for_(n, 0, n < n_avg, n + 1):
    assign(k, qua_rng.rand_int(24))
    ...

# NOT replayable: whether the draw happens depends on a measurement outcome.
with if_(state == 1):
    assign(k, qua_rng.rand_int(24))

# NOT replayable: the number of draws depends on runtime data.
with while_(counter < threshold):
    assign(x, qua_rng.rand_fixed())
    ...
```

If some draws are needed only on the controller, the host can still step over them
with `py_rng.skip(n)`, as long as `n` is known.

## QUA ↔ Python

| QUA (`qm.qua.Random`) | Python (`qiskit_qm_provider.random.Random`) | Returns |
|---|---|---|
| `Random(seed)` | `Random(seed)` | |
| `rng.set_seed(seed)` | `rng.set_seed(seed)` | |
| `rng.rand_int(max_int)` | `rng.rand_int(max_int)` | `int` in `[0, max_int)` |
| `rng.rand_fixed()` | `rng.rand_fixed()` | [`FixedPoint`](apidocs/stubs/qiskit_qm_provider.FixedPoint.rst) in `[0, 1)` |
| — | `rng.rand_int_sequence(max_int, n)`, `rng.rand_fixed_sequence(n)` | `n` ordered draws |
| — | `rng.skip(n)` | advances `n` draws |
| — | `rng.state`, `rng.draws`, `rng.initial_seed`, `rng.copy()` | inspection |

`rand_fixed()` returns a `FixedPoint`, like every fixed-valued output of the module.
Use `float(x)` to convert it, or `np.asarray(values, dtype=float)` for a list.

## Declaring the QUA generator from Python

To avoid any ambiguity about the starting state, let the Python generator create its
QUA twin. Inside `with program():`, `py_rng.declare_qua()` returns a `qm.qua.Random`
seeded with the Python generator's **current** state, so the next draw matches on both
sides. From there, the QUA instance is yours to use like any other QUA `Random`.

```python
from qm.qua import assign, declare, program
from qiskit_qm_provider.random import Random

py_rng = Random(1234)

with program() as prog:
    qua_rng = py_rng.declare_qua()   # QUA Random with the same state
    k = declare(int)
    assign(k, qua_rng.rand_int(24))

# Host side, the same draw:
k_host = py_rng.rand_int(24)
```

`declare_qua()` raises `RuntimeError` outside a QUA program. Calling it twice in the
same program declares two independent QUA generators with the same state; mirror each
one with its own `py_rng.copy()`.

Avoid constructing an unseeded QUA `Random()`: QUA picks its seed on the host at
program-build time and does not expose it. An unseeded Python `Random()` also draws a
seed with the standard library, but keeps it in `initial_seed`.

## Gaussian sampling (Box-Muller)

QUA's `Random` only draws uniformly distributed values. As a convenience, the module
includes a utility that turns uniform `rand_fixed()` draws into samples from a Gaussian
distribution with a given mean and standard deviation, using the Box-Muller transform.

[`rand_gauss_box_muller`](apidocs/stubs/qiskit_qm_provider.random.rand_gauss_box_muller.rst)
is a QUA macro that draws one `rand_fixed()` per dimension and turns it into a pair of
Gaussian samples `z1`, `z2`, using lookup tables for `sqrt(-2 ln u)` and `cos`.
[`box_muller_pair`](apidocs/stubs/qiskit_qm_provider.random.box_muller_pair.rst)
replays one macro call on the host with the same fixed-point operations, and
[`BoxMullerTables`](apidocs/stubs/qiskit_qm_provider.random.BoxMullerTables.rst) is
the single source of the lookup values for both.

The loop around it, like reseeding, stays with the user. For example, to draw
`n_samples` (even) Gaussian vectors:

```python
from qm.qua import declare, fixed, for_, program
from qiskit_qm_provider.random import BoxMullerTables, Random, box_muller_pair, rand_gauss_box_muller

py_rng = Random(seed)

with program() as prog:
    qua_rng = py_rng.declare_qua()
    mu = declare(fixed, value=mean_values)
    sigma = declare(fixed, value=std_values)
    z1 = declare(fixed, size=len(mean_values))
    z2 = declare(fixed, size=len(mean_values))
    tables = BoxMullerTables.get().declare_qua()   # declare the tables once
    b = declare(int)
    with for_(b, 0, b < n_samples, b + 2):
        rand_gauss_box_muller(qua_rng, mu, sigma, z1, z2, tables_qua=tables)
        ...  # use z1 as sample b and z2 as sample b + 1

# Host replay, same order:
samples = np.zeros((n_samples, len(mean_values)))
for b in range(0, n_samples, 2):
    z1_host, z2_host = box_muller_pair(py_rng, mean_values, std_values)
    samples[b] = np.asarray(z1_host, dtype=float)
    samples[b + 1] = np.asarray(z2_host, dtype=float)
```

Pass `mean`/`std` to `box_muller_pair` as the values the controller actually holds.
Floats are quantized with `FixedPoint(...)`, which rounds to the nearest 4.28 value
exactly as QUA's `declare(fixed, value=...)` does. Pass `FixedPoint` values directly if
you already hold the controller's exact fixed values.

Each pair costs a single uniform draw per dimension, so it has limited resolution: the
top 9 bits of the draw select one of 512 radii, and the next 9 bits one of 512 angles.

## Algorithm and provenance

The generator is a 28-bit linear congruential generator:

```
state <- (137939405 * state + 12345) mod 2**28      # advanced before each output
rand_fixed() = state / 2**28
rand_int(max_int) = (state * max_int) >> 28
```

Because 32-bit wrap-around is a multiple of `2**28`, only `seed mod 2**28` affects the
following draws.

The update rule runs in the controller's firmware; the QUA client only emits an opaque
`random` library call on the seed variable, and the QUA package does not document
the algorithm. The constants and formulas above were checked bit for bit against the
IQCC "arbel" machine (September 2026): `rand_int`, `rand_fixed` and the Box-Muller
pairs all match. That check also showed that QUA rounds to nearest when declaring
`fixed` values, and that fixed-point multiplication floors, which is what
`FixedPoint` implements.

The test suite keeps two opt-in hardware checks that stream controller draws and
compare them with the host. Run them when changing QOP versions:

- `QM_RANDOM_HARDWARE_TEST=1` with `QUAM_STATE_PATH` uses the local QuAM state;
- `QM_RANDOM_IQCC_BACKEND=arbel` fetches the latest state of an IQCC machine into a
  temporary folder, leaving `QUAM_STATE_PATH` untouched.

## Towards Qiskit integration

The [random draws design note](classical_effect_instructions.md) describes how a
Qiskit circuit could draw controller-side random values through reserved *random token*
inputs handed out by a named `Random`, and how this host mirror would then replay
those draws.
