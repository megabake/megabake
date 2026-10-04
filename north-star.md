# MegaBake North Star

MegaBake turns a captured PyTorch inference graph into a planned, composable CUDA megakernel. PyTorch and Hugging Face remain the model, parameter, and runtime interface; MegaBake owns graph-region fusion, device-body selection, scheduling, and execution planning.

## Compilation path

1. TorchDynamo captures the model invocation as an FX graph and supplies its specialization guards.
2. TorchInductor runs its post-grad FX transformations, including normalization and functionalization.
3. At a version-pinned handoff before Inductor lowering and scheduling, MegaBake receives the still-FX graph and its shape, layout, and guard metadata. This handoff is an owned integration with Inductor internals, not an assumed stable public API.
4. MegaBake performs FX-aware pattern matching and fusion, dividing the graph into regions with explicit inputs, outputs, dependencies, state effects, layouts, and numerical requirements.
5. Each supported region maps to a device-callable CuTe DSL body. MegaBake plans compatible tiling, CTA work assignment, storage lifetimes, synchronization, and stage ordering, then composes the selected bodies into one persistent CUDA kernel where legal.

## Composition rule

An FX region describes tensor work; it is not itself a GPU kernel. A region can join the megakernel only when its implementation exposes device-side work that can execute inside the enclosing kernel and satisfies the enclosing plan's data ownership, synchronization, and resource constraints. Separately launched kernel wrappers do not become composable merely because they were generated from CuTe DSL.

## Correctness and performance

The original captured graph remains the semantic reference. MegaBake preserves graph guards, outputs, mutation and state updates, aliasing, and numerical behavior through every fusion. The selected megakernel must pass complete-step correctness checks and be measured end to end against an equivalent `torch.compile` baseline for the same model, inputs, precision, and GPU. Support and performance are reported for explicit workload and GPU targets; they are not inferred for unmeasured shapes or architectures.

The north star is a small, reliable compiler path from normalized FX regions to a measured CuTe megakernel—not a requirement that every FX operation or every model fit one.


# MegaBake Compiler Architecture & Design Decisions

## Status

**Purpose:** Working architecture for MegaBake, a PyTorch-to-CUDA
megakernel compiler.

**Primary goal:** Take a live PyTorch FX/GraphModule produced during
`torch.compile`, identify computations that can execute together, and
generate efficient CuTe/CUDA kernels that reduce kernel-launch and
intermediate-memory overhead --- eventually including persistent
megakernels.

**Core architectural principle:**

> Use PyTorch/Inductor for graph preparation and normalization where it
> is valuable, but make MegaBake responsible for kernel composition, GPU
> ownership, synchronization, and megakernel scheduling.

------------------------------------------------------------------------

# 1. Problem Definition

MegaBake is not simply an FX-to-CuTe code generator.

The interesting problem is:

``` text
PyTorch computation graph
        ↓
understand computation
        ↓
identify executable regions
        ↓
decide what can execute together
        ↓
assign GPU ownership / tiling
        ↓
manage intermediate data
        ↓
manage synchronization
        ↓
generate one or more efficient CUDA kernels
```

Normal compiler stacks mostly optimize around the question:

> Which operations should belong to the same kernel?

MegaBake eventually needs to answer a stronger question:

> Which computations that would normally be separate CUDA kernel
> launches can safely and profitably become parts of the same GPU
> execution program?

That distinction is the reason MegaBake should not simply become another
thin wrapper around Inductor's scheduler.

------------------------------------------------------------------------

# 2. High-Level Architecture

The intended architecture is:

``` text
                         PyTorch
                            │
                            ▼
                    torch.compile(...)
                            │
                            ▼
                    ┌──────────────┐
                    │ FX /        │
                    │ GraphModule  │
                    └──────┬───────┘
                           │
                           │ optional/useful
                           │ Inductor post-grad
                           │ normalization
                           ▼
                 ┌────────────────────┐
                 │  MegaBake Graph    │
                 │                    │
                 │  Ops               │
                 │  Tensors           │
                 │  Dependencies      │
                 │  Metadata          │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Kernel / Region    │
                 │ Formation          │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Execution Plan     │
                 │                    │
                 │ CTA ownership      │
                 │ tiling             │
                 │ temporaries        │
                 │ synchronization    │
                 │ launch strategy    │
                 └─────────┬──────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ CuTe Backend│
                    └──────┬──────┘
                           │
                           ▼
                    CUDA Megakernel
```

A more conceptual separation is:

``` text
FX
 │
 │ What does the model compute?
 ▼
MegaBake Ops
 │
 │ What can execute together?
 ▼
Kernel Plan
 │
 │ Who owns the data and when is it visible?
 ▼
GPU Execution Schedule
 │
 │ How is it implemented?
 ▼
CuTe / CUDA
```

------------------------------------------------------------------------

# 3. Frontend: Use the Live FX Graph

## Decision

MegaBake should initially be a `torch.compile` backend.

Conceptually:

``` python
torch.compile(model, backend=megabake)
```

The compiler should consume the live `GraphModule` rather than trying to
reconstruct the graph from a textual dump.

FX already gives the compiler:

-   ordered nodes
-   node operation type
-   target
-   arguments
-   keyword arguments
-   users/dependencies
-   metadata
-   tensor information where available

Therefore:

``` text
Python source
    ↓
Dynamo / FX
    ↓
GraphModule
    ↓
MegaBake
```

There is no reason to build a Python AST parser for this purpose.

The `print_readable()` / `.txt` graph dumps are useful for inspection
and debugging only.

They should not become the compiler artifact.

------------------------------------------------------------------------

# 4. `torch.export`

`torch.export` is useful because it provides a more compiler-oriented
representation, including graph signatures and parameter/state mappings.

However, it should not be a hard requirement for the first
implementation.

Preferred initial integration:

``` text
torch.compile
    ↓
live FX GraphModule
    ↓
MegaBake
```

Potential later path:

``` text
torch.export
    ↓
ExportedProgram
    ↓
GraphModule + graph signature
    ↓
MegaBake
```

The important point is to preserve the `ExportedProgram` / graph
signature when available rather than relying only on a printed graph.

This becomes especially important for:

-   parameters
-   buffers
-   state
-   aliasing
-   input/output mapping
-   deployment-oriented compilation

------------------------------------------------------------------------

# 5. Inductor: What to Reuse

Inductor has already solved a large amount of compiler infrastructure
that MegaBake should not unnecessarily reproduce.

Useful Inductor functionality includes:

-   decompositions
-   normalization
-   canonicalization
-   post-grad transformations
-   dependency analysis
-   shape/indexing infrastructure
-   layout reasoning
-   existing PyTorch compiler behavior
-   baseline kernel generation
-   existing GEMM/template infrastructure

Therefore the intended relationship is:

``` text
FX
 │
 ▼
Inductor post-grad / normalization
 │
 ▼
MegaBake IR
```

Inductor should be treated as a **useful frontend/compiler preparation
layer**.

------------------------------------------------------------------------

# 6. Inductor Scheduler IR: Do Not Make It the Foundation

This is a deliberate architectural decision.

Do **not** make MegaBake's fundamental representation:

``` text
FX
 ↓
Inductor SchedulerNode
 ↓
MegaBake
```

Instead:

``` text
FX
 ↓
Inductor normalization
 ↓
MegaBake IR
```

Reasons:

1.  Inductor's scheduler is designed around its own backend/kernel
    scheduling model.
2.  Its scheduling abstraction is optimized around forming and ordering
    kernels.
3.  MegaBake needs to reason about persistent execution and cross-kernel
    composition.
4.  Inductor's internal scheduler IR is not a stable public compiler
    API.
5.  Depending directly on private scheduler classes would make MegaBake
    highly coupled to PyTorch internals.
6.  MegaBake needs concepts that do not naturally belong in ordinary
    kernel scheduling:
    -   CTA ownership
    -   persistent execution
    -   explicit synchronization boundaries
    -   cross-kernel dataflow
    -   intermediate residency
    -   megakernel execution stages

A thin adapter around Inductor internals is acceptable where useful, but
MegaBake should immediately translate information into its own
representation.

------------------------------------------------------------------------

# 7. Why MegaBake Needs Its Own Representation

Consider:

``` text
GEMM → SiLU
```

Ordinary compilation might produce:

``` text
kernel 1: GEMM
    ↓
global kernel boundary
    ↓
kernel 2: SiLU
```

MegaBake wants:

``` text
one kernel:

GEMM → SiLU
```

The kernel boundary normally gives CUDA a global
synchronization/visibility guarantee.

Removing that boundary means MegaBake now has to reason about:

-   who produced the data
-   who consumes the data
-   which CTA owns the data
-   whether the consumer can run immediately
-   whether the producer output can remain in registers/shared memory
-   whether another CTA needs the result
-   whether a global barrier is required
-   whether such a barrier is legal/practical
-   whether the fused version damages GEMM performance

These are MegaBake-specific execution concerns.

------------------------------------------------------------------------

# 8. Initial MegaBake IR

Do not build a huge compiler IR initially.

Start with three fundamental abstractions:

``` text
Tensor
Op
Kernel
```

Conceptually:

``` python
Tensor(
    shape=...,
    stride=...,
    dtype=...,
    device=...,
    layout=...
)
```

``` python
Op(
    kind=...,
    inputs=[...],
    outputs=[...],
    attributes={...},
    fx_nodes=[...]
)
```

``` python
Kernel(
    inputs=[...],
    outputs=[...],
    ops=[...]
)
```

For:

``` python
y = torch.silu(x @ W + bias)
```

the initial representation can simply be:

``` text
Kernel
 ├── Matmul
 ├── Add
 └── SiLU
```

This is enough to get a first end-to-end compiler working.

------------------------------------------------------------------------

# 9. Do Not Introduce a Complicated "Stage" Abstraction Initially

The word "stage" is useful later, but it should not drive the initial
design.

A stage can simply be understood as:

> A group of operations that are intended to execute under the same
> ownership/synchronization regime.

For example:

``` text
Stage 0:
    GEMM
    Add
    SiLU
```

and later:

``` text
Stage 1:
    Reduction
```

But initially this can simply be:

``` python
Kernel(
    ops=[
        Matmul(...),
        Add(...),
        SiLU(...)
    ]
)
```

Only introduce an explicit `Stage` abstraction when actual scheduling
problems require it.

The natural evolution is:

``` text
Ops
 ↓
Kernel
 ↓
Stages (only when needed)
 ↓
Execution Plan
```

Do not design a sophisticated stage hierarchy before having real
kernels.

------------------------------------------------------------------------

# 10. FX Lowering

Use a simple target-to-lowering dispatch table.

Conceptually:

``` python
LOWERINGS = {
    aten.mm.default: lower_mm,
    aten.add.Tensor: lower_add,
    aten.mul.Tensor: lower_mul,
    aten.silu.default: lower_silu,
    ...
}
```

Then:

``` text
FX node
   ↓
lowering function
   ↓
MegaBake Op
```

Examples:

``` text
aten.mm
    ↓
MatmulOp

aten.add
    ↓
AddOp

aten.silu
    ↓
SiLUOp
```

The lowering layer should convert:

``` text
Aten semantics → MegaBake semantics
```

It should NOT directly convert:

``` text
Aten → CuTe source code
```

That separation is important.

------------------------------------------------------------------------

# 11. Tensor Metadata

MegaBake tensors need enough information to generate correct kernels.

Initially preserve:

``` text
shape
stride
dtype
device
layout
```

Later potentially add:

``` text
symbolic dimensions
alignment
contiguity
aliasing
memory space
storage identity
```

Do not invent a completely independent layout system immediately.

Reuse PyTorch/Inductor layout information where possible, and introduce
MegaBake-specific layout concepts only when required by execution
planning.

------------------------------------------------------------------------

# 12. Shape-Only Operations

Not every FX operation should become a CUDA kernel.

For example:

``` text
reshape
view
transpose
permute
```

may be representable entirely through tensor metadata/indexing.

Prefer:

``` text
reshape
 ↓
permute
 ↓
GEMM
```

becoming effectively:

``` text
updated tensor metadata
        ↓
GEMM
```

rather than creating extra kernels.

This is one of the simplest early compiler wins.

------------------------------------------------------------------------

# 13. Operator Coverage

Do not try to support all of Aten.

Initial target set should be deliberately small.

Useful first operators:

``` text
placeholder
get_attr

add
sub
mul
div

relu
silu
sigmoid
gelu

mm
bmm
matmul

sum
mean

reshape
view
transpose
permute

output
```

Then expand based on actual model traces.

For transformer workloads, normalization, attention, residual, and
GEMM-related operators should receive priority.

------------------------------------------------------------------------

# 14. Region / Kernel Formation

The first fusion mechanism should be simple.

Given:

``` text
A → B → C
```

ask:

``` text
Can A and B execute together?
Can B and C execute together?
```

For example:

``` text
GEMM → Add → SiLU
```

may become:

``` text
one Kernel:
    GEMM
    Add
    SiLU
```

Do not initially build a sophisticated global optimizer or cost model.

Use simple legality rules first.

------------------------------------------------------------------------

# 15. Fusion Legality

The most important eventual fusion question is not merely:

> Are these operations adjacent?

It is:

> Can the consumer safely execute using the producer's output under the
> chosen GPU ownership and synchronization model?

For example:

``` text
GEMM
 ↓
SiLU
```

can often be CTA-local:

``` text
CTA 0:
    GEMM(tile 0)
       ↓
    SiLU(tile 0)

CTA 1:
    GEMM(tile 1)
       ↓
    SiLU(tile 1)

CTA 2:
    GEMM(tile 2)
       ↓
    SiLU(tile 2)
```

This is an ideal first fusion case.

A producer/consumer pair is much harder when the consumer requires a
global view of data produced by many CTAs.

------------------------------------------------------------------------

# 16. CTA Ownership Is a Core MegaBake Concept

Eventually every important computation needs an execution ownership
model.

For example:

``` text
GEMM output
    ↓
CTA 0 owns tile 0
CTA 1 owns tile 1
CTA 2 owns tile 2
...
```

If a consumer has compatible ownership:

``` text
producer ownership == consumer ownership
```

then the producer and consumer may be composed directly.

This is the conceptual basis for many useful fusions.

Ownership should therefore eventually become explicit in the execution
plan, rather than being hidden entirely inside backend code generation.

------------------------------------------------------------------------

# 17. Synchronization Is a Core Legality Constraint

This is one of the most important parts of MegaBake.

Normally:

``` text
kernel A
   ↓
kernel boundary
   ↓
kernel B
```

gives global ordering/visibility.

A megakernel removes that boundary.

Therefore MegaBake needs to distinguish at least:

``` text
CTA-local dependency
cross-CTA dependency
global dependency
```

The initial implementation should focus on:

``` text
CTA-local dependency
```

because it avoids difficult global synchronization.

Later, explicit barriers can be introduced.

------------------------------------------------------------------------

# 18. Global Synchronization

A persistent kernel cannot simply assume:

``` text
Stage A completes everywhere
        ↓
Stage B starts everywhere
```

unless an actual synchronization mechanism provides that guarantee.

Cooperative grid synchronization can provide a global barrier, but it
imposes launch/residency constraints.

Therefore:

**Do not make cooperative-grid synchronization a prerequisite for the
first MegaBake implementation.**

First prove CTA-local compositions.

Then experiment with global synchronization only when a real model
region requires it.

------------------------------------------------------------------------

# 19. First Execution Model

The first megakernel execution model should be:

``` text
CTA
 │
 ├── operation A
 │
 ├── operation B
 │
 └── operation C
```

rather than:

``` text
all CTAs
    ↓
global barrier
    ↓
all CTAs
    ↓
global barrier
```

This makes the first implementation much easier and naturally captures:

-   GEMM + epilogue
-   GEMM + activation
-   GEMM + residual
-   local elementwise chains
-   some normalization compositions

------------------------------------------------------------------------

# 20. Execution Plan

Eventually MegaBake needs an execution-plan representation.

Conceptually:

``` python
KernelPlan(
    inputs=[...],
    outputs=[...],

    operations=[
        Matmul(...),
        Add(...),
        SiLU(...),
    ],

    mappings={
        ...
    },

    temporaries=[
        ...
    ],

    synchronization=[
        ...
    ],
)
```

The execution plan should answer:

> Exactly how will this computation execute on the GPU?

This is the representation that separates the compiler's semantic
operations from backend-specific code generation.

------------------------------------------------------------------------

# 21. Separate Computation From Execution

This should be a hard architectural rule.

### Computation

``` text
Matmul
Add
SiLU
RMSNorm
Reduction
Attention
```

### Execution

``` text
tile size
CTA mapping
warp mapping
shared memory
register usage
pipeline
scratch storage
synchronization
persistent loop
```

Do not immediately put things such as:

``` text
tile_m = 128
num_warps = 4
```

inside the semantic `MatmulOp`.

Those are execution/backend decisions.

------------------------------------------------------------------------

# 22. CuTe Backend

CuTe should be treated as the backend/code-generation layer.

Architecture:

``` text
MegaBake IR
     ↓
Execution Plan
     ↓
CuTe lowering
     ↓
CuTe Python/JIT
     ↓
CUDA
```

Do not make CuTe syntax itself the compiler IR.

Avoid:

``` text
FX
 ↓
generate strings containing CuTe Python
```

Instead:

``` text
FX
 ↓
MegaBake semantic representation
 ↓
CuTe backend
```

This keeps the compiler architecture independent from one particular
code-generation technology.

------------------------------------------------------------------------

# 23. CuTe Code Generation Strategy

Initially, use hand-authored CuTe templates.

For example:

``` text
MatmulOp
   ↓
known CuTe GEMM implementation
```

and:

``` text
SiLUOp
   ↓
known elementwise implementation
```

Then compose them.

The first objective is not to generate arbitrary CuTe programs.

The first objective is to prove:

``` text
FX
 ↓
MegaBake IR
 ↓
known CuTe implementation
 ↓
correct CUDA kernel
```

Once that works, generalize the backend.

------------------------------------------------------------------------

# 24. GEMM Is Performance-Critical

MegaBake must not assume that fewer kernel launches automatically means
faster execution.

A fused kernel can lose if its GEMM is significantly worse than a highly
optimized GEMM implementation.

Compare:

``` text
Option A:

excellent GEMM
    ↓
kernel launch
    ↓
epilogue
```

against:

``` text
Option B:

MegaBake GEMM + epilogue
```

The fused version must eventually consider:

``` text
GEMM throughput
launch overhead
HBM traffic
L2 traffic
intermediate materialization
register pressure
shared-memory pressure
occupancy
```

The first implementation should prioritize correctness and simple
fusion, but performance evaluation must treat GEMM quality as a
first-class constraint.

------------------------------------------------------------------------

# 25. Intermediate Memory

One major reason to fuse is to avoid materializing intermediates.

Bad:

``` text
GEMM
 ↓
write intermediate to HBM
 ↓
launch
 ↓
read intermediate
 ↓
SiLU
 ↓
write result
```

Preferred where possible:

``` text
GEMM accumulator
       ↓
SiLU
       ↓
final HBM write
```

Eventually MegaBake should reason about:

``` text
register
shared memory
global memory
scratch/workspace
```

But this should evolve from concrete kernel requirements rather than
being fully designed upfront.

------------------------------------------------------------------------

# 26. Unsupported Operators

Unsupported operations should be explicit.

For example:

``` text
UnsupportedOp(...)
```

Initially:

``` text
supported region
    ↓
MegaBake

unsupported region
    ↓
reject or fallback
```

Do not silently generate an incorrect or incomplete kernel.

Later, graph partitioning can allow:

``` text
MegaBake kernel
    ↓
unsupported operation
    ↓
Inductor/PyTorch kernel
    ↓
MegaBake kernel
```

but graph partitioning should not be an initial requirement.

------------------------------------------------------------------------

# 27. Debugging Artifacts

The compiler should dump multiple levels of representation.

Suggested pipeline:

``` text
01_fx.txt
02_normalized.txt
03_megabake_ir.txt
04_kernel_plan.txt
05_generated_cute.py
06_compile.log
```

Ideally preserve IDs that connect:

``` text
FX node
   ↓
MegaBake Op
   ↓
Execution-plan operation
   ↓
generated kernel code
```

This becomes extremely important for large transformer graphs.

------------------------------------------------------------------------

# 28. Current Model Trace Findings

The available traces give two useful targets.

## SmolLM-135M

Approximately:

-   3.3k operator calls
-   211 `mm`
-   60 `bmm`
-   30 layers
-   hidden width around 576
-   MLP width around 1536
-   decomposed normalization and attention operations

This is the preferred first substantial model because its structure is
familiar and it has fewer unusual operations.

## Qwen3.5-0.8B

Approximately:

-   24.6k operator calls
-   187 `mm`
-   138 `bmm`
-   18 convolutions
-   18 `cumsum`
-   more than 1,100 `select_scatter`
-   more than 1,100 `slice_scatter`

Its structure includes both linear-attention and full-attention
components.

This makes it an excellent later stress test, especially for:

-   state updates
-   scans
-   convolution
-   scatter semantics
-   unusual dependencies

It should not be the first compiler target.

------------------------------------------------------------------------

# 29. Important Trace Limitation

The current traces are CPU traces from a fixed workload.

They use approximately:

``` text
device = CPU
batch = 1
sequence length = 4
use_cache = False
```

They also return logits for all four positions.

Therefore these traces are excellent for:

-   graph inspection
-   operator inventory
-   lowering design
-   dependency analysis

but they do not yet describe the final CUDA inference workload.

For LLM serving, separate compilation/capture paths will eventually be
needed for:

``` text
prefill
decode
KV-cache usage
different batch sizes
different sequence lengths
```

Do not let the initial `[1, 4]` trace become an accidental architectural
assumption.

------------------------------------------------------------------------

# 30. First Model Strategy

Recommended progression:

``` text
1. Toy elementwise graph
       ↓
2. GEMM
       ↓
3. GEMM + epilogue
       ↓
4. RMSNorm
       ↓
5. Transformer MLP
       ↓
6. Attention subgraph
       ↓
7. One SmolLM transformer block
       ↓
8. Whole SmolLM
       ↓
9. Qwen3.5
```

The first important milestone is not whole-model compilation.

It is:

``` text
FX
 ↓
GEMM + SiLU
 ↓
one correct CuTe CUDA kernel
 ↓
one kernel launch
```

That vertical slice validates almost every foundational layer.

------------------------------------------------------------------------

# 31. Suggested First Fusion Experiments

Good first experiments:

``` text
GEMM → bias
GEMM → SiLU
GEMM → bias → SiLU
GEMM → residual
```

Then:

``` text
RMSNorm → GEMM
GEMM → RMSNorm
```

Then larger regions.

For each experiment compare:

``` text
PyTorch eager
torch.compile / Inductor
MegaBake
```

and record:

``` text
latency
kernel count
HBM traffic
L2 traffic
GEMM throughput
register usage
occupancy
```

------------------------------------------------------------------------

# 32. Cost Model

Do not build a complicated cost model initially.

First establish:

1.  correctness
2.  simple fusion legality
3.  working CuTe kernels
4.  real benchmark results

Then introduce a cost model that can decide between:

``` text
separate optimized kernels
```

and:

``` text
fused MegaBake kernel
```

Eventually it should consider:

``` text
launch overhead
+
intermediate memory traffic
+
GEMM degradation
+
register pressure
+
occupancy
+
synchronization cost
+
workspace cost
```

The compiler should be allowed to say:

> Do not fuse.

A megakernel is not automatically better.

------------------------------------------------------------------------

# 33. Kernel Count Is Not the Objective

The objective is not:

``` text
minimize number of kernels
```

It is:

``` text
minimize end-to-end execution time
```

A five-kernel implementation can beat a one-kernel implementation if the
one-kernel implementation destroys:

-   GEMM efficiency
-   occupancy
-   memory locality
-   parallelism
-   scheduling efficiency

Therefore kernel fusion is an optimization decision, not a correctness
goal.

------------------------------------------------------------------------

# 34. Persistent Megakernel

Persistence should be introduced incrementally.

Initial:

``` text
one launch
    ↓
multiple operations
```

Later:

``` text
one persistent kernel
    ↓
multiple computation phases
    ↓
possibly repeated work / model execution
```

Do not make "persistent" a mandatory property of every first MegaBake
kernel.

First prove kernel composition.

Then introduce persistence where it provides a measurable benefit.

------------------------------------------------------------------------

# 35. Synchronization Roadmap

Recommended progression:

``` text
Level 1
CTA-local producer → consumer
```

Then:

``` text
Level 2
explicit local/shared-memory synchronization
```

Then:

``` text
Level 3
multi-CTA dependencies
```

Then, if necessary:

``` text
Level 4
cooperative-grid synchronization
```

The later levels should only be introduced in response to actual model
structures.

------------------------------------------------------------------------

# 36. Repository / Runtime Considerations

Before serious kernel compilation, ensure the CUDA toolchain is
internally consistent.

The explored environment showed:

``` text
PyTorch: 2.14.0 + CUDA 13.0
CuTe DSL: 4.8.0
GPU: H200 MIG
local nvcc: CUDA 12.8
```

The repository setup expects CUDA Toolkit 13.0.

The toolkit mismatch should be resolved before using kernel compilation
results as meaningful performance evidence.

Also preserve:

-   PyTorch revision
-   CuTe DSL version
-   CUDA toolkit version
-   GPU/MIG configuration
-   reproducible capture script

when turning experiments into tracked compiler work.

------------------------------------------------------------------------

# 37. Testing Strategy

Three levels of testing are recommended.

## Level 1: Lowering

Test:

``` text
FX → MegaBake IR
```

for individual operations.

## Level 2: Kernel correctness

Test:

``` text
MegaBake IR → CuTe → CUDA
```

against PyTorch reference results.

Use numerical comparisons such as:

``` python
torch.testing.assert_close(...)
```

## Level 3: Model correctness

Test:

``` text
PyTorch model
      ↓
MegaBake
```

against the reference model.

Only after correctness is stable should benchmark comparisons become the
main focus.

------------------------------------------------------------------------

# 38. Compilation Pipeline

The practical initial implementation should look like:

``` text
torch.compile(model, backend=megabake)
            │
            ▼
       GraphModule
            │
            ▼
    optional Inductor
    post-grad normalization
            │
            ▼
       FX lowering
            │
            ▼
       MegaBake Ops
            │
            ▼
      simple fusion
            │
            ▼
       Kernel object
            │
            ▼
      execution planning
            │
            ▼
        CuTe backend
            │
            ▼
       CUDA compilation
            │
            ▼
       executable kernel
```

------------------------------------------------------------------------

# 39. Concrete Initial API Shape

A rough API can be:

``` python
def megabake_backend(gm, example_inputs):
    graph = normalize(gm, example_inputs)

    ir = lower_to_megabake_ir(graph)

    kernels = form_kernels(ir)

    plans = plan_execution(kernels)

    compiled = generate_cute(plans)

    return compiled
```

The exact API can evolve.

The architectural boundaries should remain.

------------------------------------------------------------------------

# 40. What NOT to Build Yet

Avoid these initially:

``` text
No Python AST parser
No complete Aten coverage
No huge custom compiler framework
No MLIR dependency unless a concrete need appears
No dependence on Inductor SchedulerNode as the core IR
No generic automatic CuTe program synthesizer
No global synchronization system
No sophisticated fusion cost model
No whole-model persistent kernel
No Qwen3.5-first implementation
No generalized graph partitioner
```

Each of these can become relevant later.

None is required to prove the core idea.

------------------------------------------------------------------------

# 41. The Central Research Question

The core MegaBake research/engineering question is:

> **How can normally separate GPU kernels be composed into one efficient
> execution program while preserving or improving the performance of
> their individual high-performance implementations?**

This breaks down into:

``` text
1. Fusion legality
2. CTA/data ownership
3. Intermediate residency
4. Synchronization
5. GEMM preservation
6. Scheduling
7. Memory traffic
8. Persistent execution
```

This is the part that differentiates MegaBake from simply writing
another PyTorch backend.

------------------------------------------------------------------------

# 42. Final Architecture

The recommended final conceptual architecture is:

``` text
                         PyTorch
                            │
                            ▼
                      torch.compile
                            │
                            ▼
                         FX Graph
                            │
                            ▼
                Inductor post-grad passes
                 (normalization/decomposition)
                            │
                            ▼
                  ┌──────────────────┐
                  │   MegaBake IR    │
                  │                  │
                  │ Tensor           │
                  │ Op               │
                  │ Kernel           │
                  └────────┬─────────┘
                           │
                           ▼
                    Region formation
                           │
                           ▼
                    Fusion analysis
                           │
                           ▼
                  ┌──────────────────┐
                  │ Execution Plan   │
                  │                  │
                  │ ownership        │
                  │ tiling           │
                  │ temporaries      │
                  │ memory residency │
                  │ synchronization  │
                  └────────┬─────────┘
                           │
                           ▼
                    CuTe backend
                           │
                           ▼
                      CUDA kernel
                           │
                           ▼
                     GPU execution
```

The most important boundary is:

``` text
                 PyTorch / Inductor
                         │
                         │
              "What computation?"
                         │
                         ▼
                    MegaBake IR
                         │
                         │
              "How should it execute?"
                         │
                         ▼
                  MegaBake Plan
                         │
                         │
              "How do I implement it?"
                         │
                         ▼
                       CuTe
```

------------------------------------------------------------------------

# 43. Immediate Action Plan

The next implementation should be intentionally narrow.

### Step 1

Repair the current capture path so that the live phase-6 `GraphModule`
is passed into MegaBake instead of merely being dumped and returned.

### Step 2

Capture SmolLM on CUDA, not only CPU.

### Step 3

Build an operator inventory from the live graph.

### Step 4

Implement:

``` text
Tensor
Op
Kernel
```

as the initial MegaBake IR.

### Step 5

Implement lowering for:

``` text
mm
add
silu
reshape/view/permute
```

### Step 6

Implement one hand-authored CuTe GEMM.

### Step 7

Implement one hand-authored CuTe epilogue.

### Step 8

Make:

``` text
FX:
    mm → add → silu

MegaBake:
    one Kernel
        ├── GEMM
        ├── Add
        └── SiLU

CuTe:
    one CUDA kernel
```

### Step 9

Compare:

``` text
Inductor baseline
vs
MegaBake separate kernels
vs
MegaBake fused kernel
```

### Step 10

Only after this works, introduce explicit ownership, synchronization,
execution plans, and more sophisticated stage/region abstractions.

------------------------------------------------------------------------

# Final Design Principles

1.  **FX is the frontend; do not parse Python AST.**
2.  **Use Inductor's useful normalization/decomposition machinery.**
3.  **Do not make Inductor Scheduler IR MegaBake's permanent
    foundation.**
4.  **Keep MegaBake IR small initially.**
5.  **Use `Tensor`, `Op`, and `Kernel` first.**
6.  **Do not prematurely build a complex Stage abstraction.**
7.  **Treat CuTe as a backend, not the compiler IR.**
8.  **Separate semantic computation from GPU execution decisions.**
9.  **Make CTA ownership explicit once fusion becomes nontrivial.**
10. **Treat synchronization as a fusion legality constraint.**
11. **Start with CTA-local compositions.**
12. **Do not assume fewer kernels means better performance.**
13. **Protect GEMM performance aggressively.**
14. **Avoid intermediate HBM materialization whenever possible.**
15. **Start with GEMM + epilogue.**
16. **Start model-scale experimentation with SmolLM.**
17. **Use Qwen3.5 as a later stress test.**
18. **Keep unsupported operations explicit.**
19. **Make every compiler stage inspectable/debuggable.**
20. **Let real kernels force the IR to evolve rather than designing the
    entire IR upfront.**

The end goal is not "FX → CuTe."

It is:

``` text
                    FX
                     │
                     ▼
             compiler semantics
                     │
                     ▼
              MegaBake planning
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       ownership   memory     sync
          │          │          │
          └──────────┼──────────┘
                     ▼
               GPU execution
                     │
                     ▼
                    CuTe
                     │
                     ▼
               CUDA megakernel
```

That is the architecture that gives MegaBake room to be a real
**megakernel compiler**, rather than becoming a CuTe code generator
sitting on top of FX.
