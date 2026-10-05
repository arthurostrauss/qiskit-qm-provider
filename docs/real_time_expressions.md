# Real-Time Classical Expressions (Qiskit fork)

Some of the most useful QUA patterns rely on **classical data that is only known at run time**: a loop whose length is streamed in from the host, or a sequence of gate indices that changes from one shot to the next. Upstream Qiskit can express real-time scalar variables (`expr.Var`), but not dynamic loop bounds or arrays.

To close that gap, the author maintains a Qiskit branch that extends the real-time classical expression system:

> **Branch:** [`arthurostrauss/qiskit` @ `full_real_time_expr`](https://github.com/arthurostrauss/qiskit/tree/full_real_time_expr)

These changes are **not merged upstream** and are not in any Qiskit release. This provider works with stock Qiskit 2.x; the features on this page **only switch on when the fork is installed**.

## What the branch adds

| Feature | API | Typical use on QOP |
|---|---|---|
| **Dynamic `for`-loop ranges** | `expr.Range(start, stop, step)` as a `ForLoopOp` index set, with a real-time `expr.Var` loop counter | Noise amplification (gate folding), sweeps where the length comes from an input stream or OPNIC |
| **1-D classical arrays** | `types.Array(element_type, size)`, indexed with `expr.index(array, i)` | Gate-index sequences (randomized benchmarking), lookup tables, `ParameterVector` streamed as one array |
| **`for`-loop capture fix** | Loop bodies with a `Var` counter can read and write variables from the enclosing scope | Required by both features above: the loop body almost always reads an outer input |

The branch also adds `Expr.substitute` / `QuantumCircuit.substitute_vars`, `UnrollForLoops` support for constant `expr.Range`, OpenQASM 3 export of ranges and arrays, and **QPY version 18** to serialize all of the above. The full list is in the branch's release notes under `releasenotes/notes/`.

## Installation

The branch includes Rust code, so installing from source needs a [Rust toolchain](https://rustup.rs/) (the build takes several minutes):

```bash
pip install "qiskit @ git+https://github.com/arthurostrauss/qiskit.git@full_real_time_expr"
```

For development, clone it and install it in editable mode:

```bash
git clone -b full_real_time_expr https://github.com/arthurostrauss/qiskit.git
pip install -e ./qiskit
```

The branch reports itself as a Qiskit `2.x` development version, so it meets this provider's `qiskit<3` requirement. Install it **after** `qiskit-qm-provider` so that pip does not replace it with the PyPI release.

Check that the fork is active:

```python
from qiskit.circuit.classical import expr, types

assert hasattr(types, "Array") and hasattr(expr, "Range"), "full_real_time_expr is not installed"
```

```{note}
Circuits are lowered to QUA through OpenQASM 3 and **qm-qasm**. Array-typed `input` declarations need a qm-qasm build with input-array support (qm-qasm 1.8.0 or later).
```

## Dynamic loop ranges with `expr.Range`

A Python `range` is fixed when the circuit is built. An `expr.Range` takes **classical expressions** for `start`, `stop` and `step`, so its bounds can be real-time variables. The counter is a real-time `expr.Var`, generated automatically when you use `for_loop` as a context manager. A `Parameter` cannot be the counter, because it cannot hold a value that is only known at run time.

`expr.Range` follows **OpenQASM 3** semantics, not Python's: **both `start` and `stop` are inclusive**, and `step` defaults to 1. `expr.Range(0, 5)` runs six times (`0, 1, …, 5`) and exports unchanged as `[0:1:5]`. A Python `range(0, 5)` runs five times and exports as `[0:4]`. Two idioms give exactly `n` iterations (literals shown bare for brevity; type them narrowly, as explained in the note below):

- `expr.Range(1, n)`: the counter runs `1 … n`, and the loop is empty when `n = 0`. Use this when the counter value doesn't matter.
- `expr.Range(0, expr.sub(n, 1))`: the counter runs `0 … n-1`, as needed for array indexing. Make sure `n ≥ 1`. In Qiskit's `Uint` semantics, `0 - 1` wraps around. QUA integers are signed, so on hardware the loop is simply empty, but the circuit should not depend on that.

```{important}
The loop counter takes the range's type. A bare Python literal such as `0` is lifted to `Uint(64)`, but qm-qasm only compiles unsigned integers up to 31 bits wide. **Give the start value an explicit narrow type**, for example `expr.Range(expr.lift(0, types.Uint(16)), n)`, or build the range only from `Var`s of a narrow type.
```

### Example: noise amplification by gate folding

Zero-noise extrapolation amplifies noise by replacing a gate $G$ with $G (G^\dagger G)^n$. With a dynamic range, a **single compiled circuit** covers every fold count. The host (or an OPNIC-connected classical processor) chooses `n_fold` at run time, so there is no need to compile one program per noise level.

```python
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical import expr, types

zne = QuantumCircuit(2, name="zne_bell")
n_fold = zne.add_input("n_fold", types.Uint(8))

zne.h(0)
zne.cx(0, 1)
with zne.for_loop(expr.Range(expr.lift(1, types.Uint(8)), n_fold)):   # 1..n_fold inclusive: n_fold CX·CX pairs
    zne.cx(0, 1)
    zne.cx(0, 1)
zne.measure_all()
```

The loop is exported to OpenQASM 3 as `for uint[8] _ in [1:1:n_fold]` and compiles to `for_(i, 1, i <= n_fold, i + 1)` in QUA. It runs exactly `n_fold` times, and not at all when `n_fold = 0`.

```{warning}
At the default optimization level, `transpile` cancels each adjacent `CX·CX` pair, which leaves an empty loop and silently removes the noise amplification. Transpile folded circuits with `optimization_level=0`, or otherwise protect the folded gates from cancellation.
```

Bind `n_fold` like any other real-time input:

```python
from qiskit_qm_provider import ParameterTable, InputType

zne_inputs = ParameterTable.from_qiskit(zne, input_type=InputType.INPUT_STREAM, name="zne_inputs")
```

Use `InputType.OPNIC` instead to drive the fold count from an OAS/QUARC classical processor (see [Installation — OAS and QUARC](installation.md#open-acceleration-stack-oas-and-quarc)).

### Compile-time vs. run-time ranges

- A **constant** range, such as `expr.Range(expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8)))`, can be materialized at transpile time. `Range.values()` converts it to the equivalent Python `range` (here `range(0, 5)`, which is inclusive of 4), and `UnrollForLoops` can unroll it.
- A **non-constant** range (its bounds contain a `Var`) is kept as a loop for the hardware. `UnrollForLoops` skips it by default. Pass `strict=True` to raise an error instead.

## Classical arrays with `types.Array`

`types.Array(element, size)` is a 1-D array of `Bool`, `Uint`, `Float` or `Duration`. You can declare one with `add_input` (streamed from the host) or `add_var` (local to the circuit). Read or write an element with `expr.index(array, i)`, where the index may be a real-time expression such as a loop counter. Writes go through `qc.store(expr.index(arr, i), value)`.

When the fork is installed, two things change in this provider:

- [`ParameterTable.from_qiskit`](apidocs/stubs/qiskit_qm_provider.parameter_table.ParameterTable.rst) maps an `Array` input to **one array-valued `Parameter`** (a QUA array of `int`, `bool` or `fixed`). Earlier provider versions misread it as a scalar `fixed`.
- A `ParameterVector` is collapsed into **one** array `Parameter` named after the vector, in place of the `_name_i_` scalars used with stock Qiskit. The fork's OpenQASM 3 exporter emits a matching `input array[float[64], N]`. To keep the old unrolled layout, pass `qiskit_supports_array=False`.

## Worked example: two-qubit randomized benchmarking

Two-qubit RB applies a random sequence of two-qubit Cliffords, followed by the Clifford that inverts the whole sequence, and measures the survival probability as a function of depth. In plain Qiskit this means **one circuit per random sequence**. A typical experiment (around 10 depths × 30 sequences) then compiles 300 QUA programs, or one very long program.

With arrays and dynamic ranges, a **single circuit** interprets any sequence:

1. On the host, each Clifford is lowered to a short list of **op codes** from a small native alphabet. Here the alphabet is `{sx, rz(π/2), cz}` on two qubits, which generates the full two-qubit Clifford group.
2. The op codes are streamed into an `Array(Uint(8), MAX_OPS)` input, and the sequence length into a `Uint` input.
3. On the device, a `for` loop over `expr.Range(0, n_ops - 1)` (both ends inclusive, with a narrow-typed start) reads `ops[i]` and dispatches it with a `switch`.

The loop body reads the outer input `ops`, which the capture fix in the branch makes possible.

### The circuit

```python
import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.classical import expr, types

MAX_OPS = 512                     # capacity of the on-device sequence buffer
SX0, SX1, S0, S1, CZ = range(5)   # op codes; S_k := rz(pi/2) on qubit k

rb = QuantumCircuit(2, name="rb_2q")
meas = ClassicalRegister(2, "meas")
rb.add_register(meas)

ops = rb.add_input("ops", types.Array(types.Uint(8), MAX_OPS))
n_ops = rb.add_input("n_ops", types.Uint(16))

start = expr.lift(0, types.Uint(16))                 # narrow type: qm-qasm needs uint width <= 31
last = expr.sub(n_ops, 1)                            # inclusive stop: i = 0 … n_ops-1 (n_ops >= 1)
with rb.for_loop(expr.Range(start, last)) as i:       # i: real-time Uint(16) loop counter
    with rb.switch(expr.index(ops, i)) as case:       # dynamic array indexing
        with case(SX0):
            rb.sx(0)
        with case(SX1):
            rb.sx(1)
        with case(S0):
            rb.rz(np.pi / 2, 0)
        with case(S1):
            rb.rz(np.pi / 2, 1)
        with case(CZ):
            rb.cz(0, 1)

rb.measure([0, 1], meas)
```

The exported OpenQASM 3 shows the two new constructs:

```text
input array[uint[8], 512] ops;
input uint[16] n_ops;
...
for uint[16] _loop_i_0 in [0:1:n_ops - 1] {
  switch_dummy = ops[_loop_i_0];
  switch (switch_dummy) { case 0 { sx q[0]; } ... case 4 { cz q[0], q[1]; } }
}
```

### Host side: sampling and encoding sequences

```python
from qiskit import transpile
from qiskit.quantum_info import Clifford, random_clifford


def encode(circuit: QuantumCircuit) -> list[int]:
    """Lower a 2-qubit Clifford circuit to op codes from {sx, rz(pi/2), cz}."""
    native = transpile(circuit, basis_gates=["sx", "rz", "cz"], optimization_level=1)
    codes = []
    for inst in native.data:
        q = native.find_bit(inst.qubits[0]).index
        if inst.name == "sx":
            codes.append(SX0 + q)
        elif inst.name == "rz":   # Clifford angles are multiples of pi/2
            codes += [S0 + q] * (round(float(inst.params[0]) / (np.pi / 2)) % 4)
        elif inst.name == "cz":
            codes.append(CZ)
    return codes


def rb_sequence(depth: int, rng) -> list[int]:
    total = Clifford(QuantumCircuit(2))
    codes = []
    for _ in range(depth):
        clifford = random_clifford(2, seed=rng)
        codes += encode(clifford.to_circuit())
        total = total.compose(clifford)
    codes += encode(total.adjoint().to_circuit())   # recovery Clifford
    return codes
```

A depth-20 sequence encodes to about 300 op codes, so `MAX_OPS = 512` is enough for depths up to about 30. Increase it for longer sequences. Only the first `n_ops` entries are read, so pad the rest with zeros.

### QUA side: one program for all sequences

```python
from qm.qua import program, declare, for_, save, stream_processing
from qiskit_qm_provider import ParameterTable, InputType

tqc = transpile(rb, backend)   # switch-case bodies are already native
rb_inputs = ParameterTable.from_qiskit(tqc, input_type=InputType.INPUT_STREAM, name="rb_inputs")
# rb_inputs holds: ops -> QUA int array[512], n_ops -> QUA int

n_sequences, n_shots = 300, 1000

with program() as rb_prog:
    rb_inputs.declare()
    seq = declare(int)
    shot = declare(int)
    backend.init_macro()

    with for_(seq, 0, seq < n_sequences, seq + 1):
        rb_inputs.rcv()   # host pushes a new (ops, n_ops) pair
        with for_(shot, 0, shot < n_shots, shot + 1):
            comp = backend.quantum_circuit_to_qua(tqc, rb_inputs)
            save(comp.outputs.state_ints["meas"], comp.outputs.streams["meas"])
            # (or accumulate survival counts in real time)
```

On the host, push each sequence while the job is running:

```python
rng = np.random.default_rng(1234)
for depth in depths:
    for _ in range(sequences_per_depth):
        codes = rb_sequence(depth, rng)
        rb_inputs.push_to_opx(
            {"ops": codes + [0] * (MAX_OPS - len(codes)), "n_ops": len(codes)},
            job=job,
        )
```

Compared with the batch-of-circuits approach, the program is compiled **once**, the FPGA holds a single sequence buffer, and new random sequences cost only one input-stream transfer. The same pattern applies to interleaved RB (add an op code for the target gate), cycle benchmarking, and any other protocol driven by a random gate sequence. With `InputType.OPNIC`, a classical co-processor can generate the sequences on the fly.

## Compatibility notes

- **Stock Qiskit:** none of the APIs on this page exist. `ParameterTable.from_qiskit` falls back to unrolled scalar fields for `ParameterVector`s, and circuits without arrays or `expr.Range` compile exactly as before.
- **QPY:** circuits that use `expr.Range`, arrays or `Var` loop counters need QPY version 18. They cannot be loaded by upstream Qiskit.
- **Array input cost:** qm-qasm currently binds each `input` array by declaring its own QUA array and copying the streamed values element by element. The copy runs at the start of **every** `quantum_circuit_to_qua` call site, so inside a shot loop it costs about `MAX_OPS` QUA assignments per shot, and the array is held twice on the FPGA. Keep `MAX_OPS` no larger than you need.
- **Limits:** arrays are 1-D only, with no nesting and no slices. The element types are `Bool`, `Uint`, `Float` and `Duration`.

## Related

- [Parameter Table](parameter_table.md): real-time inputs, `from_qiskit`, `filter_function`
- [Backend & Utilities](backend.md): `quantum_circuit_to_qua` and hybrid embedding
- [Workflows](workflows.md#4-hybrid-quaqiskit-programs-embedding-circuits-in-qua)
