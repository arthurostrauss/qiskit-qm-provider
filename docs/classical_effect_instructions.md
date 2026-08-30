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
| `SetRandomSeed` | typed unsigned-integer expression | updates the selected RNG state | `rng.set_seed(seed)` |
| `RandInt` | maximum typed integer expression | writes a `uint` variable | `assign(target, rng.rand_int(maximum))` |
| `RandFixed` | no value input | writes a floating-point variable | `assign(target, rng.rand_fixed())` |

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

The first implementation should have one named default generator per compiled
QUA program. `SetRandomSeed` changes that generator; each later draw advances
the same stream in circuit order. This avoids the misleading alternative of
constructing `Random(seed)` at every `RandInt`, which can re-seed each draw.

The implementation and documentation must state whether a seed is applied at
program entry, on every shot, or when an explicit `SetRandomSeed` instruction
is reached inside a controller loop. Those choices have different
reproducibility and sampling behaviour.

## Intended OpenQASM and QUA lowering

The preferred exported form is a classical assignment whose right-hand side is
an opaque provider function call, not a gate call and not an explicit `defcal`
definition:

```
uint[32] sample;
qm_set_random_seed(seed);
sample = qm_rand_int(16);
```

At QUA lowering time, this becomes the equivalent of:

```python
rng = Random()
rng.set_seed(seed)
assign(sample, rng.rand_int(16))
```

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
  with typed expression validation and clear copy semantics.
- Add `QuantumCircuit` helpers that accept or create a destination `expr.Var`.
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
- Introduce a **per-compilation lowering context** that owns the QUA `Random`
  object. A backend-global macro closure is not sufficient: it could retain a
  QUA variable from a previous compilation or declare the generator inside an
  incorrect branch scope.
- Declare the generator once in the intended outer QUA scope, then share it
  among all random operations in that compilation.

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
- Provide a per-`compile` context or lifecycle hook in which one QUA `Random`
  instance can be created and reused safely throughout code generation.
- Add integration tests for typed assignment, generator reuse, reseeding, and
  repeated compilation with the same backend/compiler instance.

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
- Test deterministic sequences, explicit reseeding, and no cross-compilation
  leakage of generator state.
- Document the numerical contract, generator scope, reproducibility, and
  performance/latency assumptions separately from hardware results.

## Recommended implementation sequence

1. **Write the semantic contract first.** Fix the default-generator lifetime,
   seed timing, integer width/range, fixed-point mapping, and error behaviour.
2. **Prototype the qm-qasm boundary.** Compile a handwritten OpenQASM
   assignment with an opaque zero-qubit function call and verify it returns a
   QUA value into a classical variable. This resolves the main uncertainty
   before adding provider API surface.
3. **Add the per-compilation RNG context.** Choose whether it belongs in the
   provider's compilation wrapper or a small qm-qasm extension; do not hide it
   in a persistent Target macro closure.
4. **Implement `SetRandomSeed` and `RandInt`.** Keep the first slice integer
   only and explicit-destination only.
5. **Add `RandFixed` after numeric semantics are validated.** Its QUA
   fixed-point behaviour deserves separate tests and documentation.
6. **Only then assess convenience syntax or upstream work.** Returning an
   `expr.Var` from a helper is compatible with this design; returning an
   effectful expression is not a small provider extension.

## Decision record

The recommended first implementation is an explicit-destination,
provider-specific classical-effect instruction family lowered as opaque
OpenQASM function calls inside ordinary classical assignments. It requires
targeted provider work and likely a small qm-qasm lifecycle extension, but it
does not require a full Qiskit classical-expression redesign or explicit
OpenQASM `defcal` support.
