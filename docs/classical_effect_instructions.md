# Design note: controller-side random draws in Qiskit circuits

```{important}
This is a future implementation note, not a public API commitment. Code that uses the
proposed API is marked **(proposed)**; every other snippet runs against the current
provider, Qiskit and qm-qasm. It records the chosen semantics, the evidence behind them, and
the alternatives that were rejected.
```

## Motivation

A QUA program can draw pseudorandom numbers on the controller with `qm.qua.Random`. A
circuit could then use those values for random Clifford selection, dithering, or a
randomized feedback policy, without a host round-trip. Qiskit has no notion of such a
value. Its classical expressions (`qiskit.circuit.classical.expr`) are pure, whereas a
random draw advances the state of a generator.

The provider already ships [`qiskit_qm_provider.random.Random`](random.md), a host mirror
of QUA's generator that reproduces its draws when their number and order are known. The
goal is for that one object to play two roles:

- **authoring**: the circuit obtains its random values from it;
- **host twin**: it replays those same values on the host.

## The model: a draw is a read of a random token

A random draw is written as an ordinary Qiskit `Store` whose right-hand side is a
**random token**. A token is a reserved input variable that the circuit declares but never
gives a value to. Each execution of that `Store` makes one draw from the QUA generator
the token names.

```python
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical import expr, types

qc = QuantumCircuit(1, 1)
token = expr.Var.new("__qm_rand_int__rb__0__max_24", types.Uint(5))
qc.add_input(token)
clifford = qc.add_var("clifford", expr.lift(0, types.Uint(5)))

qc.store(clifford, token)  # one draw in [0, 24) from the generator "rb"
with qc.if_test(expr.equal(clifford, 3)):
    qc.x(0)
```

This exports as plain OpenQASM 3:

```
input uint[5] __qm_rand_int__rb__0__max_24;
uint[5] clifford;
clifford = 0;
clifford = __qm_rand_int__rb__0__max_24;
if (clifford == 3) {
  x q[0];
}
```

Every stage handles only standard objects:

- the circuit holds only a `Var` and `Store`s, with no new instruction and no new
  expression node;
- the transpiler sees ordinary `Store`s, which carry proper DAG wires;
- the exporter emits an ordinary input declaration and an ordinary assignment.

Only the final compilation to QUA knows that reading the token means "draw".

### Token contract

1. **Name grammar.** `__qm_rand_<kind>__<rng>__<n>__<spec>`, where:
   - `<kind>` is `int` or `fixed`;
   - `<rng>` is the name of the host `Random`. It is unique within a program and contains no `__`;
   - `<n>` is a counter kept by that `Random`, so tokens stay unique across circuits and
     after `compose`;
   - `<spec>` is one of:

     | `<spec>` | Meaning | QUA draw |
     |---|---|---|
     | `max_<k>` | integer in `[0, k)`, `k` a literal | `rng.rand_int(k)` |
     | `max_v_<name>` | integer in `[0, name)`, `name` a circuit input | `rng.rand_int(<value of name>)` |
     | `unit` | fixed-point number in `[0, 1)` (`kind` = `fixed`) | `rng.rand_fixed()` |

   The name is self-describing. A circuit loaded from QPY, or read in another process,
   can be compiled and replayed from the circuit and the seed alone.
2. **Type.** The token has the type of its destination:
   - `Uint(k)` with `1 <= k <= 31` and `2**k >= max`;
   - `Float` for `fixed`.

   qm-qasm maps `uint[k]` to a signed QUA `int` and restricts `k` to 31 bits, so `Uint(32)`
   is not allowed. With matching types, the `Store` never needs a cast.
3. **Use.** A token appears **only as the whole right-hand side of exactly one `Store`**,
   and never as an lvalue. Draw sites are therefore exactly those `Store` sites, and no
   Qiskit pass can duplicate or drop a draw by rewriting an expression. A provider check
   enforces this before export.
4. **Placement.** The draw `Store` may appear anywhere, including under
   measurement-dependent control flow or inside loops. The draw is then made at run time
   wherever that `Store` executes. Whether the host can *predict* the value is a separate
   question (see [Host replay](#host-replay)).
5. **Variable ranges.** In `max_v_<name>`, `name` must be a circuit input that the
   circuit never writes. This is required for ordering (see
   [Ordering and the DAG](#ordering-and-the-dag)). A range computed inside the circuit
   must wait for a later milestone. A range given as an expression, such as `2*n + 1`,
   is supplied as an input already holding that value.

## Authoring API (proposed)

The host `Random` gains a `name` and hands out tokens. Helpers append the `Store`, so
users never spell token names:

```python
from qiskit_qm_provider.random import Random

rng = Random(seed=1234, name="rb")                     # (proposed) name argument

qc = QuantumCircuit(1)
nmax = qc.add_input("nmax", types.Uint(5))
a = qc.add_var("a", expr.lift(0, types.Uint(5)))
b = qc.add_var("b", expr.lift(0, types.Uint(5)))
noise = qc.add_var("noise", expr.lift(0.0, types.Float()))

qc.rand_int(a, max=24, rng=rng)        # (proposed) __qm_rand_int__rb__0__max_24
qc.rand_int(b, max=nmax, rng=rng)      # (proposed) __qm_rand_int__rb__1__max_v_nmax
qc.rand_fixed(noise, rng=rng)          # (proposed) __qm_rand_fixed__rb__2__unit
```

A helper takes a new token from `rng` and adds it with `qc.add_input` if the circuit does
not already hold it. It then appends `qc.store(dest, token)`. The helper validates:

- the destination type;
- `max >= 1`;
- the width bound;
- that a variable range is a circuit input.

Random values can be used wherever Qiskit accepts a `Var`: conditions, `switch` targets,
`Store`s, indices, and
[`ConditionalPlay`](backend.md) conditions. They cannot be gate angles directly, because
Qiskit gate parameters are `Parameter`s, not `Var`s.

## Compilation

### The two execution modes

- **Embedded (primary).** The user writes the QUA program, owns its shot loop, and
  calls `backend.quantum_circuit_to_qua(qc, ...)` inside it. The user also owns the QUA
  generator and passes it in:

  ```python
  with program() as prog:
      qua_rng = rng.declare_qua()                       # declared once; see below
      with for_(n, 0, n < shots, n + 1):
          backend.quantum_circuit_to_qua(tqc, inputs=..., random={rng: qua_rng})  # (proposed)
  ```

- **Standalone (visualisation).** Called outside a program, the provider opens
  `with program():` itself, declares `rng.declare_qua()`, and compiles inside it.
  Without this, qm-qasm would turn every unprovided input into an input stream, and the
  draws would be lost. This mode is useful mainly to inspect the generated QUA.

### Binding tokens: the one qm-qasm change

For every circuit input whose name starts with `__qm_rand_`, the provider:

1. decodes the name;
2. finds the QUA generator in `random` by the `<rng>` field;
3. binds the name to a factory that builds the draw expression.

`ParameterTable.from_qiskit` and the input-declaration path skip that prefix, so a
token is never declared as a streamed parameter.

qm-qasm needs one small, generic addition: an input bound to an expression that is
evaluated **on every read**. Today, a value provided in `inputs` is copied into a newly
declared variable once, where the input is declared
(`code_generation_transformation.py`, `_declare_variable`). With that copy, a draw
inside a loop would reuse the same value on every iteration. The proposed public marker is
`qm_qasm.InputExpression(factory)`:

- `_declare_variable` declares nothing for it. It registers a read-only entry in the
  variables database whose `qua_value_exp` returns `factory(lookup)` each time it is
  accessed. Every identifier read goes through that property
  (`parse_expression.py`, `variables_db[signature].qua_value_exp`).
- `lookup(name)` returns the QUA value of a circuit symbol. This is how `max_v_<name>`
  reads its range at the moment of the draw.
- A `Store` into the token fails, because the entry has no `assign`.

This is about ten lines plus tests. It touches no grammar, no validator and no
function-call dispatch, and nothing in it is specific to random numbers. The provider
binds tokens as follows:

```python
InputExpression(lambda lookup: qua_rng.rand_int(24))              # max_24
InputExpression(lambda lookup: qua_rng.rand_int(lookup("nmax")))  # max_v_nmax
InputExpression(lambda lookup: qua_rng.rand_fixed())              # unit
```

A prototype of the marker was monkeypatched into qm-qasm, and the embedded example
above then generated the QUA below. The token draws appear exactly where the circuit
reads them. The token inside a measurement-dependent `if_` draws only when the branch is
taken, and the token inside a loop draws on every iteration:

```python
v1 = declare(int, value=1234)          # the user's generator state
...
with for_(v3, 0, v3 < 2, v3 + 1):      # the user's shot loop
    assign(v4, v2)                     #   m (an ordinary input copy)
    with if_(v4):
        assign(v5, Random(v1).rand_int(24))
    with for_(v6, 0, v6 <= 2, v6 + 1):
        assign(v5, Random(v1).rand_int(24))
        with if_(v5 == 3):
            play('x180', 'q0')
```

`Random(v1)` adopts the existing variable `v1` as its state and declares nothing new, so
all draws advance the same stream.

(randomness-ownership-and-seed-timing)=
## Randomness ownership and seed timing

The generator state never enters the circuit. It lives in the user's QUA program, and
seed timing is simply wherever the user places `declare_qua()` and `set_seed()`.

- **Once per program, by default.** QUA hoists every `declare(..., value=v)` to the top of
  the program and applies `v` once, even when the declaration sits inside a loop. So a
  generator declared once with `rng.declare_qua()` persists across shots.
- **One stream per program.** Passing the same `qua_rng` to several
  `quantum_circuit_to_qua` calls makes all those circuits draw from one stream, in
  execution order. The provider's own job builder (`job/qua_programs.py`) declares its
  generators before its shot loop.
- **Reseeding** is the user's `qua_rng.set_seed(...)`, mirrored by `rng.set_seed(...)` on
  the host, exactly as in the [replication contract](random.md).

This matters because qm-qasm lowers circuit-level initialisation into per-execution
assignments:

- `uint[28] rb = 7;` becomes a hoisted declaration plus an `assign(rb, 7)` in the loop body;
- a provided input is copied into a fresh variable on every execution.

A generator state kept *inside* the circuit would be restarted, or silently copied, on every shot.

(ordering-and-the-dag)=
## Ordering and the DAG

The only ordering information a `DAGCircuit` holds is its wires. Qiskit adds variable
wires only for control-flow operations (their condition or target, and the variables their
blocks capture) and for `Store` (its lvalue and rvalue), in `additional_wires`
(`crates/circuit/src/dag_circuit.rs`). A custom `Instruction` whose `params` contain a
`Var` gets **no** variable wire.

Every transpilation, even at optimization level 0, converts the circuit to a DAG and
back. The final circuit is materialised by `lexicographical_topological_sort` with the
sort key `(qubits, clbits)`. So a node that touches no qubit or clbit has the smallest key
and is emitted as early as its incoming edges allow.

Random tokens are safe because a draw is a `Store`: it is wired on its destination and on
the token. Two independent draws on the same generator share no wire and may be swapped by
the transpiler. That changes nothing statistically, and host replay follows the order of
the *transpiled* circuit (see [Host replay](#host-replay)). A variable range is read
inside the qm-qasm binding, not by the `Store`, so the DAG cannot see that read. That is
why the range must be an input the circuit never writes.

The same analysis exposes an existing bug in `ConditionalPlay`. Its Boolean condition is
held in `params`, so its reads are not wired. After a single
`circuit_to_dag`/`dag_to_circuit` round-trip, a later `Store` to the condition variable
moves ahead of the play, and the play then reads the overwritten value:

```python
from qiskit.converters import circuit_to_dag, dag_to_circuit
import qiskit_qm_provider  # registers QuantumCircuit.conditional_play

qc = QuantumCircuit(1)
v = qc.add_var("v", expr.lift(False))
qc.x(0)
qc.store(v, expr.lift(True))
qc.conditional_play("x180", v, 0)      # intended to read True
qc.store(v, expr.lift(False))
print([i.name for i in dag_to_circuit(circuit_to_dag(qc)).data])
# ['store', 'store', 'store', 'x', 'qm_conditional_play_x180_…']  -> reads False
```

**Fix (planned):** `qc.conditional_play` wraps the instruction in a `box`. The control-flow
builder captures the variables found in the instruction's parameters, so the box node is
wired to the condition's variables and clbits as well as to its qubit. qm-qasm's
`visit_Box` inlines the body, so the QUA output is unchanged.

(host-replay)=
## Host replay

Host replay is a separate concern from the controller semantics above. A provider
analysis pass walks the transpiled circuit, lists its draw sites in execution order, and
classifies each one:

- **replayable**: the site executes a number of times the host knows. This covers
  straight-line code and loops with a known bound.
- **not replayable**: the site sits under measurement-dependent control flow or inside a
  loop whose length depends on data. The controller draws correctly; the host just cannot
  know the value without streaming it back.

For a replayable circuit, the host replays the draws from a copy of `rng` taken when
`declare_qua()` was called:

- a token `max_<k>` replays as `rand_int(k)`;
- a token `max_v_<name>` replays as `rand_int(value of name)`;
- a token `unit` replays as `rand_fixed()`.

This gives, for example, the Clifford that each shot ran, without streaming anything back.
It turns the [replication contract](random.md) into a static check.

## Required work by layer

### qm-qasm

- Add `InputExpression(factory)`, accepted in `Compiler.compile(..., inputs=...)`: no
  declaration, a read-only entry, `factory(lookup)` evaluated on each read.
- Tests: a read inside `if`, `for`, `while` and `switch` blocks; `lookup` of an input;
  a `Store` into an `InputExpression` is rejected; repeated compiles sharing one generator.

### Provider

- `Random`: a `name` argument (identifier-safe, no `__`) and a token counter.
- Helpers `QuantumCircuit.rand_int(dest, max, rng)` and `QuantumCircuit.rand_fixed(dest, rng)`.
- A token checker run before export, enforcing the contract: the grammar, a single
  `Store` right-hand side, never an lvalue, types, and variable ranges being read-only inputs.
- `quantum_circuit_to_qua(..., random=...)`: decode the tokens and bind them. Skip the prefix
  in `ParameterTable.from_qiskit` and in input declaration. Support standalone mode by
  opening the program.
- The job builder declares generators before its shot loop. The estimator and sampler
  builders forward `random` if they support it.
- The replay analysis pass.
- The `ConditionalPlay` `box` fix.

### Documentation and tests

- Unit tests: token names, helper validation, the checker, DAG round-trips that keep draws
  after the writes they depend on, and exporter text.
- An end-to-end test with an in-memory QuAM machine: transpile, compile, and compare the
  generated QUA with host replay.
- Update the [Random numbers guide](random.md) with the circuit workflow.

## Rejected alternatives

Each alternative below was checked against the current code.

- **Provider instructions exported as opaque function calls**
  (`sample = qm_rand_int(16);`). qm-qasm would inline a zero-qubit call as a macro with no
  `align`, but three steps block that path today:
  - `ReferencingValidator` rejects the undeclared function name;
  - an assignment's right-hand side accepts only the math built-ins (`NotImplementedError`);
  - a standalone call statement (`ExpressionStatement`) is a disallowed node type.

  The instructions also have no DAG wires, so the transpiler hoisted a draw above a
  measurement and a conditional reseed at optimization levels 0 and 1. A zero-qubit Target
  entry keyed on `()` also crashes `VF2Layout` at levels 2 and 3. Qiskit's qasm3 AST has no
  `FunctionCall` node, and `DefcalCallStatement` drops the parentheses when a call has no
  arguments.
- **A string-keyed registry of named modules** (`qm_rand_int("rb", 24)`). String literals
  are not valid OpenQASM 3 expressions. A registry created during compilation would also
  need a lifecycle hook that neither the provider nor qm-qasm has.
- **Generator state as a circuit variable**, advanced by `Random(state_var)` inside a
  macro. qm-qasm restarts or copies circuit state on every execution (see
  [seed timing](#randomness-ownership-and-seed-timing)). Keeping the state would need a
  new by-reference input mode, or a copy-back by the provider after every circuit.
- **A draw instruction wrapped in a `box`** for ordering. It works, but it keeps all the
  function-call costs above for no benefit over a `Store`.
- **An effectful expression node in Qiskit** (e.g. `expr.Call`). This is the most principled
  representation. It needs changes to the Qiskit expression system (types, copying,
  serialisation, DAG wiring), a `FunctionCall` exporter path, and qm-qasm call resolution.
  If Qiskit adopts such a node, tokens can be translated to it mechanically.
- **`DefcalInstruction` with a typed return.** Its base exporter accepts only `None` or
  `Bool` returns, written to a clbit. `ConditionalPlay` already overrides
  `build_defcal_call` to emit typed arguments, but a draw needs no such machinery once it
  is a `Store`.

## Implementation sequence

1. **qm-qasm `InputExpression`**, with the tests above.
2. **`Random.name` and the token helpers**, with the checker and DAG round-trip tests.
3. **Binding in `quantum_circuit_to_qua`** for embedded mode, then standalone mode and the
   job builder.
4. **The `ConditionalPlay` `box` fix.** It is independent and can land at any point.
5. **The replay analysis pass**, and replay helpers on `Random`.
6. **Deferred:** ranges computed inside the circuit (they need a DAG-visible read of the
   range, e.g. a guarding `if (m > 0)`), Gaussian draws built on `rand_fixed`, and random
   gate angles fed through `Parameter` inputs.

## Decision record

- A random draw is a `Store` from a reserved, self-describing token input
  (`__qm_rand_<kind>__<rng>__<n>__<spec>`). Each token is used as the right-hand side of
  exactly one `Store`.
- The user owns the QUA generator and passes `random={rng: qua_rng}`; one stream per program.
- A variable range must be a read-only circuit input.
- The only compiler change is a generic qm-qasm `InputExpression` input, evaluated on each read.
- Host replayability is analysed separately, on the transpiled circuit.
- `ConditionalPlay` is boxed so that its condition reads are ordered.
