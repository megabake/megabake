# MegaBake V3 architecture

Status: pipeline-first revision, 2026-09-09; proposed, not implemented or benchmarked. Start with the
[document index](MEGABAKE_V3_README.md) and [evidence audit](MEGABAKE_V3_GPU_REANALYSIS.md).

## 1. Decision

Build a generic FX-to-persistent-kernel compiler that jointly optimizes **math, tile readiness,
data movement and execution order**. Its output is a tile-level compute-and-movement program,
not a concatenation of complete operator kernels. Competitive device bodies remain necessary;
the compiler must also expose the opportunities that a sequence of opaque bodies would hide.
The first backend targets single-GPU inference with fixed shape buckets and specializes each
compatible workload/target combination.

Four mechanisms are part of the primary architecture: consumer-aligned fusion, staged loads
across task boundaries, producer/consumer tile overlap, and tail-aware work assignment. Removing
launches is an additional benefit. A phase-barrier executor is retained as a control, not the
performance architecture that must win before these mechanisms are attempted.

The research success condition is one owned compute grid per supported invocation, with correct
outputs and state, beating the strongest tested equivalent `torch.compile` baseline. A CUDA Graph
containing several kernels is useful as a baseline or fallback, but it is a different result.

One grid does not imply one thread block, one block per physical SM, one tile per worker, no global
memory, zero launch cost, or a kernel that stays alive across every generated token.

The initial invocation is **one cached autoregressive step**. A long-lived multi-token controller
is deferred. The existing uncached sequence-one forward remains a diagnostic benchmark.

## 2. The three representations

| Representation | New decisions it enables | What it must preserve |
|---|---|---|
| Normalized FX + `TensorFacts` | Functionalization, selected decomposition, shape/alias/effect analysis | Original outputs, state transitions, guards, numerical semantics |
| `SemanticGraph` | FatOp recognition, implementation-independent composites, attention/state distinctions | Executable reference semantics and all live boundary values |
| `ExecutionPlan` | Body capabilities, tile actions, reduction continuations, readiness/release tokens, transport, placement and launch configuration | SemanticGraph behavior, progress and storage safety under explicit guards |

`SemanticGraph` should initially remain an FX graph using a MegaBake operation dialect plus
ordinary ATen regions. There is no need for a second graph library or an MLIR infrastructure project.
The separation is semantic, not a demand for three storage formats.

`LayerSummary`, `TargetProfile`, candidate records, buffer allocation, resource reports, and
benchmark results are analyses or plan fields. They are not independent IRs. CUDA/CuTe is the
backend's existing machine-level representation; a general Tile IR is deferred until another
backend or transformation demonstrates a concrete need. Tile actions and movement are explicit
inside ExecutionPlan now; deferring a general instruction-level Tile IR does not defer tiling or
fine-grained scheduling.

Detailed contracts live in [IR and reuse](MEGABAKE_V3_IR_AND_REUSE_PLAN.md).

## 3. Where to reuse Inductor

Take the handoff while the program is still normalized FX/ATen, before `GraphLowering` converts
it to Inductor's loop, buffer, and external-kernel objects. Reuse selected early transformations
through one pinned adapter. Preserve matmul, norm, activation, and attention semantics whenever
possible; decompose unsupported fragments on demand.

There is no universal point described by “all Inductor optimizations are done, but fusion has not
started.” FX passes can already perform fusion; later scheduling and code generation perform
other optimizations. Inductor IR is not optimized ATen that can simply be read back into FatOps.
These distinctions follow the inspected PyTorch 2.6
[compilation flow](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/compile_fx.py),
[FX passes](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/fx_passes/post_grad.py), and
[IR definitions](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/ir.py).

The compiler should accept generic graph inputs, with an explicit supported semantic subset.
Unknown operations retain their reference region. Strict compilation rejects an unsupported
region with a diagnostic; the normal callable may use an explicitly reported compiled fallback.
Accepting generic FX does not establish one-grid support for arbitrary Python effects, devices,
data-dependent control flow, custom kernels, or unbounded dynamic shapes.

## 4. FatOps and layers

Initial FatOps are `Linear`, `RMSNorm`, `Pointwise`, `SwiGLU`, `RoPE`, and `SDPA`, together with
explicit cache/state transitions. Embedding and ordinary views remain supported primitives.
Each FatOp has an executable reference expansion, metadata/shape rules, and numerical/effect
contracts. `SwiGLU(g,u)` means the activation/gating computation; the gate/up projections are
separate linears that may form a larger composite.
Other gated activations keep their exact Pointwise reference; they must not be relabelled SwiGLU.

Keep a small menu of overlapping composite candidates: linear epilogues, gate/up plus SwiGLU,
QKV projection, norm plus linear, and RoPE plus cache write. Match first, select jointly with
tiling, transport and scheduling. A large semantic match does not force a fused device body.
Do not commit a non-overlapping semantic cover before discovering that a different cover exposes
head readiness or a useful reduction chunk. Preserve legal alternatives until plan selection.

`LayerSummary` groups repeated subgraphs and associates layer-local state. It helps reuse code
templates, tuning results, and liveness analyses. It does not collapse a layer to an opaque node.
Only promote a structured layer/loop representation when a pass needs explicit iteration/state
semantics that the summary cannot supply.

The supplied Hugging Face example has a repeated three-linear/one-full-attention pattern. Its
`config.json` is model metadata, not a GGUF-specific compiler IR. Treat it as a hint to check
against the graph. Full/windowed/causal attention describe connectivity; FlashAttention is an
algorithm for computing attention; gated recurrent attention has different state transitions.
The precise example is discussed in the [IR document](MEGABAKE_V3_IR_AND_REUSE_PLAN.md).

## 5. Target knowledge

Use one small `TargetProfile`, populated from CUDA queries and a documented architecture feature
table, then augmented by optional calibration. Separate legality from cost:

- Legality: instruction availability, launch limits, cooperative support, memory scope, alignment,
  and actual compiled resources.
- Cost: exact-shape/stage latency, joint bandwidth and compute contention, publication cost,
  usable prefetch lead time, tail behavior, and whole-entry composition penalty.

Each cost has provenance and an uncertainty/status field. Without a GPU, costs are `UNKNOWN` or
explicit priors. Hardware facts alone cannot select a performance winner.

A DFS can walk a description that already exists. It cannot discover undocumented memory
latencies, bandwidth under contention, or arbitrary chip topology. Model only the memory spaces
and movement mechanisms that affect a candidate. See [hardware model](MEGABAKE_V3_HARDWARE_MODEL.md).

## 6. Joint planning, then a bounded pipelined executor

The unit of scheduling is a legal tile action, not necessarily a complete FatOp. Actions include
weight preload, activation acquire, compute/update, publish and storage release. A body may expose
only an indivisible tile; another may expose a continued reduction or separate load/compute stages.
The planner can exploit only the capabilities a body actually supplies.

Use bounded search over a small number of body/tile choices, consumer groupings, edge transports,
worker assignments and lookahead depths. Derive dependencies from read/write/reduction footprints;
choose scratch lifetimes and synchronization in that same search. Estimate resource-constrained
makespan, then compile legal finalists and measure the complete entry. Do not select the fastest
isolated GEMM first and hope it leaves room for a pipeline. Unknown costs retain exploratory
alternatives; they cannot certify a winner.

One ExecutionPlan has two lowering policies:

- **Barrier control:** complete an operator/region, then a cooperative grid barrier. This provides
  a simple reference and a matched-body control for launch-only composition.
- **Pipelined regions:** statically generated worker/cohort programs with explicit readiness and
  bounded staging. A worker can preload its next independent weight tile while current work runs;
  consumers start on completed input regions without waiting for unrelated producer tiles.

Static assignment does not mean synchronous execution. Start with generated instruction order
and small producer/consumer cohorts, not a universal central task queue. The compiler proves that
the chosen worker order and buffer credits cannot create a wait cycle. Data-DAG acyclicity alone
is insufficient. An optional ready-task dispatcher is an extension of this plan, not a second IR.

A cooperative grid contains `P` resident worker CTAs. Logical tiles are independent of `P`; a
60-worker grid may execute thousands. `blockIdx.x` identifies a worker, not a physical SM. All
workers reach each retained grid join, even if they have no arithmetic in that region. Actual
compiled resources and cooperative-launch support bound `P` before launch.

### First-class transformer pipelines

| Region | What becomes possible | What still blocks progress |
|---|---|---|
| QKV -> RoPE/cache -> attention | Finish a head/GQA group and start its attention while other projection tiles run | All inputs for that head, cache visibility and valid lengths; not just Q readiness |
| Gate/up -> exact gating -> down | Publish hidden chunks; an output-tile owner incrementally accumulates down-projection K chunks | A chunk must be complete; final cast/residual waits for every required chunk |
| Current task -> next task/layer | Prefetch independently addressed weights into a reserved slot before next activations exist | Free staging space, descriptor validity, bandwidth and the later activation dependency |
| Ready consumer work + producer tail | Fill compatible idle-worker time instead of imposing an operator-wide barrier | Ready useful work, legal ownership, available workers and nonconflicting resources |

For down projection, prefer testing an owner-held FP32 accumulator across ordered K chunks
against full materialization. Global split-K partials plus an explicit finalizer are a separate
option when ownership/parallelism requires them, not a mandatory cost of every streamed reduction.
Both alternatives must respect the declared numerical policy. Attention-output projection can
use the same continuation mechanism only with its own supported body and footprint proof.

The detailed action/lifetime protocol, worked schedules, progress conditions and validation cases
are normative in [pipelining and scheduling](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md).

## 7. Device bodies and resource compatibility

The first linear portfolio has a coalesced warp/CTA reduction GEMV and one supported tensor-core
family. Evaluate cuBLASDx early as a candidate; use adapted CUTLASS/CuTe where its controls are
needed. Exact shapes select algorithms. `M=1` alone neither mandates SIMT nor rules out tensor cores.

Code reuse stops below a host launcher. A candidate must declare its participating threads,
shared storage, scratch lifetime, asynchronous operations, output ownership, and logical tile
interface. It additionally declares atomic-tile, preloadable and/or streamed-reduction capability.
An internal GEMM pipeline does not automatically expose a cross-operator pipeline interface.
A callable C++ function is not enough if it relies on an incompatible block or scheduler.
The [kernel reuse document](MEGABAKE_V3_KERNEL_REUSE.md) defines this boundary, including current
cuBLASDx pipeline restrictions.

Compile the complete candidate entry before deciding residency. Inlining, register liveness,
stack/local memory, instruction footprint, shared-storage overlap, and library protocols can
make separately fast bodies slow when combined. The resource report must come from the actual
linked/generated entry, not the maximum of a spreadsheet of isolated register counts.

For mutually exclusive scratch lifetimes, shared storage can be overlaid. Coexisting current and
prefetched tiles, accumulators, library stages and metadata must all fit. Allocate from proven
partial-order lifetimes/explicit releases, never from a predicted timing diagram. Distinguish
output publication from permission to overwrite an async operation's source buffer. The launch
reserves its required footprint throughout execution; small phases do not recover occupancy.

Keep only the selected reachable bodies. Preserve expensive operations that the workload truly
needs, including real decode attention; removing unrelated prefill alternatives must not remove
required attention work.

The single-grid planner explores a small number of compatible common block configurations. A
particular 128- or 256-thread choice is a candidate, not a universal architectural rule. If no
configuration both works and wins, record a strict loss. A split graph is a separately scored
fallback, not a way to meet the single-grid constraint.

## 8. Storage, fusion, and state

Represent an edge's selected transport as a view, global allocation, CTA-local forwarding, or
pure recomputation. Global address-space traffic is not automatically physical HBM traffic;
intermediates can hit L2. Use counters to substantiate physical-byte savings.

Choose tiles with consumers in mind: finish complete head groups, pair corresponding gate/up
chunks, and preserve useful weight access patterns. A layout that reduces one instruction count
but scatters the next consumer's loads can lose the end-to-end comparison. Pack stable weights
at session setup only with explicit space/setup accounting; dynamic activation conversion is timed.

Register/shared-memory forwarding requires compatible ownership and lifetime inside the same
CTA, or an explicitly supported cluster protocol. A producer finishing on one worker cannot hand
its registers or CTA shared memory to an arbitrary different worker.

RMSNorm replication is an option, not a default. Recomputing the norm per logical tile can multiply
its cost badly for a large vocabulary head. Consider computing the statistic once, retaining the
vector once per worker across its tiles, or materializing a small L2-resident vector. Compare the
complete cost. Never move the normalization scalar across a GEMV merely by assuming floating-point
distributivity across intermediate casts.

KV/state updates remain observable effects even if their returned tensor is unused. Model them
functionally with explicit new state or effect dependencies, then select in-place storage only
after proving alias/lifetime safety. Repeated calls and overlapping sessions must not race on the
same arena. Start with one serialized session; concurrency uses separate state/arenas or explicit
ownership.

## 9. Invocation and fallback

The initial runtime has a small session API: initialize stable weights/state/workspace, validate
guards, bind the step inputs, launch the selected entry, and return an output with documented
ownership. Input copies, descriptor updates, resets, and output copies count in complete latency.
The pipelined entry initializes its own event counters before a uniform grid barrier; no hidden
reset kernel or host-side counter clone is assumed free. The barrier control can omit counters
when its plan has none. All required initialization belongs in complete latency.

Preserve the normal callable's output lifetime. An explicit `run_into` or borrowed-output session
can be faster, but receives an equivalent baseline and a distinct benchmark contract. Stable
input addresses do not imply that new token data arrives for free.

Use an ordinary compiled callable as the initial fallback and a stable-address CUDA Graph as the
performance control. A custom hybrid graph planner is optional later. Dynamic vendor-node
parameter patching is deferred until binding copies are a demonstrated bottleneck.

GraCE's vendor path changes launch arguments of a separate graph node; it cannot import private
cuBLAS code into the persistent entry. This is consistent with the
[paper, §4.2.1](https://www.usenix.org/system/files/osdi26-ghosh.pdf) and
[CUDA parameter-layout API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html).

## 10. Performance contract

Select for the resource-constrained critical path of the complete invocation, not minimum kernel
count or the sum of isolated body timings. For an equivalent baseline, the practical question is
whether removed dispatch gaps, materializations, tails and exposed pipeline stalls pay for body
slowdowns and persistent coordination:

```text
saved baseline work > added body + resource + synchronization + invocation cost
```

The [performance model](MEGABAKE_V3_PERFORMANCE_MODEL.md) turns this into equations and budgets.
Its numbers are conditional examples, not V3 predictions. Analytical lower bounds can reject a
target speedup; they cannot certify an achievable implementation.

Mandatory controls separate reusable-body quality, one-grid execution, fusion, prefetch and
tile readiness. Demonstrating one compute grid is not demonstrating a pipeline benefit. Show a
non-launch improvement over a matched phase control as well as the strongest-baseline result;
reject a particular optimization when its resource/coordination cost outweighs its benefit.

The strongest tested equivalent baseline includes legal default, reduce-overhead, and max-autotune
configurations, using matched capture/input/output/state contracts. Mode names do not prove graph
capture succeeded. Selection trials and final validation samples are separate.

There is no unconditional strict-win guarantee for arbitrary FX graphs. The compiler can offer
semantic correctness and honest fallback; performance remains measured by supported cell.

## 11. First deliverable and expansion

The first deliverable is a vertical experiment, not a large compiler framework:

1. Reproduce the baseline and audit correctness for SmolLM2-135M and the actual Gemma-2B checkpoint.
2. Extract their hot shapes; compare one lean embedded GEMV and one tensor-core candidate.
3. Generate a full block with explicit state and both barrier-control and pipelined plans;
   evaluate head-ready attention, streamed MLP and cross-task weight staging in this experiment.
4. Generate whole-model cached-decode steps at batch one and at least short/long context buckets.
5. Accept a strict win only after unprofiled end-to-end measurement and complete state validation.

Use whichever suitable GPU becomes available first; the H200 records are priors, not a requirement
to obtain the same partition. Portability needs a second real target after the first useful result.

Pipelining is inside the initial vertical path; it is not gated on a static model already winning.
Once that path is measurable, extend its limiting dimension. Prefill, recurrent state,
quantization and batching are separate workload extensions. Each
addition needs an observed bottleneck or a stated new workload; none is a universal prerequisite.

## 12. Invariants

- Graph semantics do not depend on model names or GPU product strings.
- Every FatOp has a reference definition; every transformation preserves live outputs and effects.
- Only compatible device bodies enter a strict entry; no hidden external grids.
- Logical tiling and worker residency are independent.
- Resource legality is checked after compilation and before launch.
- Buffers are reused according to the selected execution order and asynchronous lifetimes.
- Publication, scratch release and reduction finalization have distinct, verified meanings.
- Wait-for progress includes worker order and resource credits, not only tensor dependencies.
- Body, tiling, fusion, transport and schedule selection remain coupled until final planning.
- Performance comparisons include the same work and numerical contract.
- Unknown measurements stay unknown; a fallback never counts as a strict win.
