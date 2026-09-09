# Design note: classical-effect instructions and controller-side random values

:::{important}
This is a future implementation note, not a public API commitment. It records
the intended semantics and compiler boundaries for provider instructions that
produce controller-side classical values.
:::

## Motivation

`ConditionalPlay` established a useful provider pattern: a Qiskit instruction
can expose a deterministic QUA capability without pretending that it is a
unitary gate or a Qiskit control-flow operation. Its Boolean condition is an
input to an instruction and can be represented by Qiskit's typed classical
expression system.

Controller-side random-number generation is a different case. A QUA `Random`
generator has mutable state and each draw produces a value that must be usable
by later circuit operations. It therefore cannot be faithfully represented as
a pure `expr.Expr`: copying, substituting, or evaluating a pure expression is
not supposed to change program state, while a random draw must advance an RNG
state.

The proposed family is consequently made of **classical-effect instructions**:
ordered circuit operations that write into a declared Qiskit classical
variable. The variable, rather than the instruction itself, is reused in
subsequent Qiskit expressions.

## Proposed circuit model

The first family would be deliberately small:

| Provider operation | Inputs | Output effect | QUA operation |
|---|---|---|---|
| `SetRandomSeed` | typed unsigned-integer expression; optional RNG name | updates the selected RNG state | `rngs[name].set_seed(seed)` |
| `RandInt` | maximum typed integer expression; optional RNG name | writes a `uint` variable | `assign(target, rngs[name].rand_int(maximum))` |
| `RandFixed` | optional RNG name | writes a floating-point variable | `assign(target, rngs[name].rand_fixed())` |

The public helpers should operate on Qiskit `expr.Var` values. A helper may
return the destination variable for convenience, but the underlying operation
is still a write, not a value-producing expression:

```python
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical import expr, types

qc = QuantumCircuit(1)
seed = qc.add_input("seed", types.Uint(32))
sample = qc.add_var("sample", types.Uint(32), 0)

qc.set_random_seed(seed)
qc.rand_int(sample, maximum=16)

condition = expr.equal(sample, expr.lift(3, types.Uint(32)))
qc.conditional_play("x180", condition, 0)
```

An `int` supplied by a user can be lifted into a provider-defined unsigned
integer type, after range checking. Runtime values should already be typed
Qiskit expressions. `RandFixed` should expose a Qiskit floating-point type,
while its documentation must retain QUA's fixed-point execution semantics and
range constraints.

### RNG lifetime is an explicit semantic choice

QUA already allows multiple independent `Random` instances in one program.
Each instance owns its own LCG state; draws and reseeds affect only that
instance. Correlated default seeds across unseeded constructors are a known
QUA footgun, so distinct user-facing generators should be seedable
independently.

The circuit model should therefore treat **named RNG modules** as first-class
resources, not assume a single ambient generator. A convenience default
(e.g. `"default"`) can still exist so simple circuits omit an explicit handle,
but that default is one entry in a per-compilation registry, not a uniqueness
constraint.

`SetRandomSeed` changes the selected generator; each later draw on that same
handle advances that stream in circuit order. This avoids the misleading
alternative of constructing `Random(seed)` at every `RandInt`, which can
re-seed each draw.

The implementation and documentation must state whether a seed is applied at
program entry, on every shot, or when an explicit `SetRandomSeed` instruction
is reached inside a controller loop. Those choices have different
reproducibility and sampling behaviour, and they apply **per named
generator**.

### Insights: many RNG modules, one compilation

This subsection records design options and open questions for multi-generator
support. It does not yet commit the public API.

**Why multiple sources matter.** Realistic controller programs often need
independent streams: Clifford selection vs. shot dither vs. a feedback-policy
noise term, or separate RNGs for disjoint qubit groups so reseeding one
experiment does not scramble another. Folding everything into one stream
forces users to manually multiplex draws and destroys reproducibility
isolation.

**Declaration vs. first use.** Two workable shapes:

1. **Explicit declare** — a zero-qubit instruction such as
   `DeclareRandomModule(name, seed=...)` (or a circuit helper
   `qc.add_random_module("rb")`) that registers the name before any draw.
2. **Lazy declare-on-first-use** — the first `RandInt`/`SetRandomSeed` that
   mentions a name materializes that generator in the lowering context.

Explicit declare is clearer for OpenQASM export, Target preflight, and
"unused name" diagnostics. Lazy declare is friendlier for small scripts but
makes name typos create silent new streams. A hybrid is attractive: helpers
may create on first use while export/lowering still require every used name to
appear in a compile-time registry.

**How the handle appears on operations.** Prefer an optional named argument on
every random classical-effect instruction, defaulting to the built-in default
module:

```python
qc.set_random_seed(seed)                      # default module
qc.rand_int(sample, maximum=16)               # default module
qc.set_random_seed(seed_rb, rng="rb")
qc.rand_int(clifford, maximum=24, rng="rb")
qc.rand_fixed(noise, rng="dither")
```

Alternatives considered and currently disfavoured for the first milestone:

- Separate instruction families per module (`RandIntRb`, …) — does not scale.
- Passing a Python `RandomModule` object token through the circuit — harder to
  serialize, copy, and remap through control-flow / QASM scopes than a stable
  string (or interned symbol) name.
- Encoding the module name only in the opaque OpenQASM function symbol
  (`qm_rand_int_rb`) without a circuit-level handle — works for export but
  weakens circuit IR clarity and makes Target registration combinatorial.

**OpenQASM shape.** Keep assignment-form opaque calls, but thread the module
identity into either the function name or a leading literal/identifier
argument. Name-suffix form is closer to today's conditional-play naming
discipline; argument form keeps one operation family and lets qm-qasm resolve
a single hardware op that looks up the generator in the compile context:

```
qm_set_random_seed("rb", seed);
clifford = qm_rand_int("rb", 24);
noise = qm_rand_fixed("dither");
```

Either encoding is fine if the provider exporter and qm-qasm agree. The
important invariant is: **function identity + module key select exactly one
QUA `Random` instance from the per-compilation map**.

**Lowering context.** Replace "own the one `Random` object" with "own a
`dict[str, Random]` (or equivalent) created once in the intended outer QUA
scope." Rules worth fixing early:

- Names are compilation-scoped symbols (stable strings); circuit copies must
  preserve them.
- Declaring the same name twice is an error unless the second declare is a
  no-op with identical seed policy.
- Using an unknown name is a hard compile/export error.
- Unseeded modules must document whether QUA's constructor-time Python seed
  is captured once per compilation (reproducible across shots of that program
  object) or re-drawn somehow — today QUA captures at program creation.
- Independent modules must not share one underlying QUA `Random`; reseeding
  `"rb"` must leave `"dither"` untouched.

**Default module policy.** Keep a single well-known default name so the
one-liner API in the examples above remains valid. Document that advanced
users should name every stream they care about; relying on the default plus
ad-hoc multiplexing is the anti-pattern this design is meant to avoid.

**Interaction with seed timing.** Seed-at-entry, seed-per-shot, and
seed-when-instruction-executes remain per-module choices. A program may mix
policies across modules only if that is explicit; the first milestone can
require one global seed-timing policy and still allow many modules.

**Suggested first milestone without blocking multi-RNG.** Implement the
registry and optional `rng=` handle from day one, even if examples only show
the default module. Retrofitting names later would churn instruction payloads,
QASM encodings, and qm-qasm context hooks. What can wait: rich declare
helpers, per-module seed-timing overrides, and any upstream Qiskit
"classical resource" abstraction beyond a string key.

**Open questions to resolve before coding the API.**

- Is the module key a free string, a provider `Enum`, or a small dedicated
  `RandomModule` marker type that stringifies for QASM?
- Must modules be declared in the circuit before use, or only registered with
  the backend/Target?
- Do we need a Qiskit-visible classical "resource" object (analogous to a
  stretch or a typed var) so DAG/transpiler passes can see RNG dependencies,
  or is instruction-carried `rng=` metadata enough?
- Caps: is there a practical upper bound on concurrent QUA `Random` objects
  we should document or enforce?
- Should `RandBit` / measure-like variants, if added later, share the same
  module namespace?

## Intended OpenQASM and QUA lowering

The preferred exported form is a classical assignment whose right-hand side is
an opaque provider function call, not a gate call and not an explicit `defcal`
definition. The default-module spelling can omit the module key in user-facing
helpers while still lowering through the same registry:

```
uint[32] sample;
qm_set_random_seed(seed);       # default module
sample = qm_rand_int(16);       # default module
```

At QUA lowering time, this becomes the equivalent of looking up (or creating)
the named entry in the per-compilation map:

```python
rngs = {}  # owned by the compilation context
rngs["default"] = Random()
rngs["default"].set_seed(seed)
assign(sample, rngs["default"].rand_int(16))
```

A second named stream is the same pattern with a different map key, not a
second compilation-global singleton.

The Qiskit provider exporter would recognize the provider instructions before
normal instruction handling and build the assignment and function-call AST
nodes directly. It should use Qiskit's typed-expression AST builder for all
arguments and destinations, as `ConditionalPlay` does for its Boolean input.

## Why `DefcalInstruction` is not the primary representation

Qiskit's `DefcalInstruction` is exporter metadata, rather than a
result-producing circuit instruction abstraction. Its current base exporter:

- treats its parameters as angle-like;
- accepts only `None` and `Bool` return types;
- models a non-void result as exactly one output `Clbit`.

It could be useful for a narrow measure-like `RandBit` experiment, but it does
not model a `uint` or floating-point result stored in an `expr.Var`.
Subclassing it would not by itself extend the circuit IR. The provider would
still need to override the QASM builder's defcal-call logic and invent an
output-location convention. That is more coupling for less expressive power
than a dedicated assignment-form exporter extension.

Making `expr.rand_int(...)` itself return a Qiskit expression is a separate,
substantially larger upstream-Qiskit project. It would require an effectful
classical-expression node, type rules, substitution and copy semantics,
serialization, DAG dependencies, and transpiler preservation. It is not
required for the explicit-destination design above.

## Required work by layer

### 1. Provider circuit API and instruction model

- Define `SetRandomSeed`, `RandInt`, and `RandFixed` provider instructions
  with typed expression validation, an optional named RNG handle, and clear
  copy semantics.
- Add `QuantumCircuit` helpers that accept or create a destination `expr.Var`
  and optionally select an RNG module (defaulting to the built-in name).
- Preserve the destination and all typed expressions through circuit copies,
  control-flow block copies, and transpilation.
- Register the operations as zero-qubit `Target` instructions. Qiskit's
  `Target` supports zero-qubit instructions, but the provider must test this
  through the exact transpilation passes it supports.
- Treat these instructions as side-effecting directives for provider purposes:
  no inverse, no quantum control, no commutation or deletion assumptions.

### 2. Provider OpenQASM exporter

- Extend `QMOpenQASM3Exporter` with a narrow dispatch for these instructions.
- Emit `ClassicalAssignment` statements with a typed `FunctionCall` right-hand
  side, rather than serializing values through numeric gate parameters.
- Map copied `expr.Var` and bit resources back into the current export scope.
- Validate that every opaque function name is registered by the backend before
  export, just as conditional plays are preflighted against the Target.

### 3. Provider-to-QUA operation mapping

- Add no-qubit operation mappings whose macros return a QUA scalar for
  `RandInt` and `RandFixed`, and mutate generator state for `SetRandomSeed`.
- Introduce a **per-compilation lowering context** that owns a map of QUA
  `Random` instances keyed by module name (including the default). A
  backend-global macro closure is not sufficient: it could retain a QUA
  variable from a previous compilation or declare generators inside an
  incorrect branch scope.
- Declare each used generator once in the intended outer QUA scope, then
  share that instance among all random operations that name it in that
  compilation.

### 4. qm-qasm compiler assessment and required changes

The proposed assignment/function-call form deliberately avoids explicit
OpenQASM `defcal` support. The current qm-qasm code already has most of the
right structural path:

- `FunctionCall` is an allowed AST node.
- Its code generator treats a non-built-in function call as a zero-qubit
  hardware operation and returns its result.
- `ClassicalAssignment` assigns that result to a declared QUA variable.

Therefore **a new parser grammar or explicit-defcal implementation should not
be required merely to accept an opaque call such as `sample =
qm_rand_int(16);`**. This must nevertheless be demonstrated with a compiler
integration test, because the OpenQASM parser, operation resolver, and
provider-generated source have to agree on the exact AST shape.

The likely qm-qasm work is narrower but still real:

- Confirm or extend resolution and configuration of a zero-qubit,
  result-producing hardware operation.
- Provide a per-`compile` context or lifecycle hook in which a map of QUA
  `Random` instances can be created and reused safely throughout code
  generation.
- Add integration tests for typed assignment, per-module generator reuse,
  independent reseeding of distinct modules, and repeated compilation with
  the same backend/compiler instance.

If the design instead emits real `defcal` definitions or relies on typed
defcal returns, **qm-qasm changes become mandatory and substantially larger**:
its allowed AST types currently exclude calibration-definition nodes, and its
code generator has no generic assigned-defcal-call lowering path. That is
outside the recommended first milestone.

### 5. Tests, documentation, and compatibility

- Test integer literals, typed unsigned inputs, dynamic maxima, and fixed
  draws; reject incompatible types and invalid limits clearly.
- Test reuse in `expr` comparisons, `if_test`, `switch`, and
  `ConditionalPlay` conditions.
- Test copies, nested control-flow blocks, layout/transpilation, and QASM
  export scope remapping.
- Test deterministic sequences, explicit reseeding, independent multi-module
  streams, and no cross-compilation leakage of generator state.
- Document the numerical contract, named-generator scope, reproducibility, and
  performance/latency assumptions separately from hardware results.

## Recommended implementation sequence

1. **Write the semantic contract first.** Fix named-module identity, the
   default-module convenience, seed timing, integer width/range, fixed-point
   mapping, and error behaviour for unknown / duplicate names.
2. **Prototype the qm-qasm boundary.** Compile a handwritten OpenQASM
   assignment with an opaque zero-qubit function call and verify it returns a
   QUA value into a classical variable. This resolves the main uncertainty
   before adding provider API surface.
3. **Add the per-compilation RNG registry.** Choose whether the
   `dict[str, Random]` (or equivalent) belongs in the provider's compilation
   wrapper or a small qm-qasm extension; do not hide it in a persistent Target
   macro closure. Exercise at least two named modules in the prototype.
4. **Implement `SetRandomSeed` and `RandInt` with optional `rng=`.** Keep the
   first slice integer only and explicit-destination only, but carry the
   module key from day one.
5. **Add `RandFixed` after numeric semantics are validated.** Its QUA
   fixed-point behaviour deserves separate tests and documentation.
6. **Only then assess convenience syntax or upstream work.** Returning an
   `expr.Var` from a helper is compatible with this design; returning an
   effectful expression is not a small provider extension. Richer
   `DeclareRandomModule` helpers can follow once the registry path is proven.

## Decision record

The recommended first implementation is an explicit-destination,
provider-specific classical-effect instruction family lowered as opaque
OpenQASM function calls inside ordinary classical assignments, with a
**per-compilation registry of named QUA `Random` modules** and a convenience
default name for simple circuits. It requires targeted provider work and
likely a small qm-qasm lifecycle extension, but it does not require a full
Qiskit classical-expression redesign or explicit OpenQASM `defcal` support.
