# MegaBake V3: reuse good math, not opaque launches

Research date: 2026-09-09. This is a source-backed design assessment; no library adapter was
implemented or benchmarked. See [research provenance](MEGABAKE_V3_RESEARCH_AND_DECISIONS.md).

## 1. The decisive boundary

The strict backend needs a computation that the threads of its existing grid can execute.
A host-callable library API, a launchable kernel handle and a device-callable tile routine are
three different interfaces.

| Source/mechanism | Useful for an external baseline? | Can supply owned in-grid math? | V3 decision |
|---|---|---|---|
| cuBLAS/cuBLASLt host call | Yes | Not through that host API | Strong baseline/fallback |
| Captured vendor `CUfunction` | Potentially, with its full launch contract | Not by treating the handle as device code | Optional tracing, not critical path |
| GraCE vendor prelude | Dynamic graph bindings | No; target remains a separate grid | Defer until binding cost matters |
| CUDA Graph/device graph launch | Low-overhead multi-grid execution | Does not inline constituent kernels | Baseline or separate hybrid result |
| Dynamic parallelism | Can launch supported child work | Child launch is still another grid | Does not satisfy strict one-grid goal |
| cuBLASDx block operations | Can be wrapped for testing | Yes, subject to descriptor contracts | Early candidate |
| cuBLASDx pipelines | Can be wrapped for testing | Potentially, with entry/pipeline adaptation | Bounded compatibility experiment |
| CUTLASS/CuTe device components | Yes | Yes, after adapting their execution contracts | Main controllable source |
| Selected MPK task bodies | MPK provides its own runtime too | Some are explicitly device-callable | Inspect/adapt individual bodies |
| Hazy staged bodies | Its demos supply full schedules too | Yes with substantial stage/resource adaptation | Reference for load/compute/store and retirement interfaces |
| Luminal kernel implementations | Yes in their own runtime | KernelOp alone is not a BlockOp | Learn mappings and contract separation |

The graph and child-grid distinctions follow NVIDIA's
[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
and [dynamic parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
execution models. One host submission is not necessarily one compute grid.

## 2. Why the cuBLAS extraction route does not solve composition

Tracing a library launch can reveal its function identity, dimensions, shared-memory request and
argument values at that launch. It can help identify a tactic and reproduce a carefully controlled
external experiment. It does not establish the lifetime of workspaces/descriptors, all hidden
dependencies, or a stable public ABI for replaying that tactic across library versions.

The [driver execution API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
uses `CUfunction` for launches and introspection. `cuFuncGetParamInfo` returns parameter offsets
and sizes, not source, device-callable code, or a writable parameter-buffer base address.

Even possessing a vendor cubin would not mechanically turn its entry into a composable routine.
Such a transformation would need to preserve or rewrite launch indexing, thread participation,
barriers, shared storage, exits, relocations, asynchronous protocols and register allocation.
A binary rewriter plus verifier is a separate research project, not a simpler way to obtain GEMM.
V3 does not claim binary transformation is impossible in principle; it rejects it as the critical
path to this compiler's next performance result.

### What GraCE actually contributes

GraCE distinguishes rewritable JIT kernels from immutable vendor kernels. Its JIT path introduces
argument indirection in code; its vendor path places a prelude before a separate vendor graph node.
This addresses replay binding costs, not in-grid code composition. See
[GraCE §4.2.1, Figures 8–9](https://www.usenix.org/system/files/osdi26-ghosh.pdf).

The paper's wording about obtaining parameter storage via `cuFuncGetParamInfo` is stronger than
that API's documented return contract. Any reproduction must resolve the concrete graph-node
update/storage mechanism and its supported-version restrictions. Do not elevate an unresolved
bridge into a stable MegaBake dependency.

V3's simpler alternative is stable session buffers and explicit bindings. If those copies later
dominate, investigate documented graph parameter updates or owned-kernel indirection. A custom
vendor-binding mechanism is not needed to test strict persistent math.

## 3. cuBLASDx: a real option, not a cuBLAS performance guarantee

Regular cuBLASDx descriptors expose block-level GEMM with shared-memory/register accumulation
interfaces. The caller still owns global tiling, loading, synchronization and output handling.
See [GEMM execution methods](https://docs.nvidia.com/cuda/cublasdx/api/gemm_methods.html).

The inspected [0.7.1 requirements](https://docs.nvidia.com/cuda/cublasdx/0.7.1/requirements_func.html)
require CUDA 13.0+, C++17 and CUTLASS 4.4.1 or newer; the package includes that CUTLASS version.
This is not a drop-in match for the historical nvcc 12.8/CUTLASS 3.8 environment. Pin a compatible
older release or deliberately test a newer isolated toolchain. Do not silently upgrade the whole
project just to make a proposed descriptor compile.

cuBLASDx supplies building blocks; it does not promise the exact private cuBLAS tactic or equal
performance for every shape. M=1 may favor a K-parallel GEMV even when a tensor-core descriptor
is legal. Conversely, padded tensor-core work can win for some large-N shapes. Measurement chooses.

### Current pipeline interface: important additional constraints

The [0.7.1 pipeline guide](https://docs.nvidia.com/cuda/cublasdx/0.7.1/using_pipelines.html) describes
global-memory staged GEMM, host-created pipeline metadata, and per-block tile objects. It requires
global dimensions divisible by descriptor tiles and pipeline depth no greater than the K-tile
count. Its device handle remains a `__grid_constant__` kernel argument; that annotation belongs
on a `__global__`, not a `__device__`, parameter. The interface exposes required block dimensions
and shared-buffer size/alignment. The completed K accumulation feeds the epilogue.

**V3 adapter hypothesis, not validated support:** generate the persistent entry with the required
typed constant parameters; create legal tile objects using logical tile coordinates rather than
assuming one tile per physical block; complete each tile's async work before reusing scratch.
Probe whether this supports repeated phases and many layer-specific bindings without illegal
copies, excessive kernel-argument storage or resource growth. Test each assumption against the
pinned headers and compiler. Do not hide these objects behind an invented type-erased device ABI.

For M=1 and a larger M tile, divisibility may require padded activations or another interface.
Charge activation padding/initialization per invocation; weight packing can be amortized only
when weights remain stable. A nominally fast padded GEMM that needs expensive surrounding work
can lose the full comparison. Never assume undocumented masked-tail support.

The first experiment may use the simpler block interface. The pipeline path earns inclusion by
demonstrating compatible execution and lower composed latency; its richer feature set alone is
not a reason to require it for every linear.

Crucially, an internally pipelined GEMM is not automatically `PRELOADABLE` across different
operators or `STREAM_REDUCTION` across newly arriving activation chunks. The adapter must prove
those interfaces separately. If a cuBLASDx call only supports a full reduction, retain it as an
atomic-tile candidate and test an adapted CUTLASS/CuTe or SIMT continuation for streamed MLP.
Do not make cuBLASDx compatibility the prerequisite for all cross-task scheduling.

## 4. CUTLASS/CuTe: reuse below the host adapter

CUTLASS separates low-level operations/tiling, collective mainloops and epilogues, kernel-level
composition, and device launch adapters. Those layers have different reuse contracts; see the
[GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html).

For V3, a collective or an adapted tile body is a better boundary than calling a host `Gemm` object.
Choose one target-supported tactic first. Retain its tested memory layout and pipeline where
possible, but make logical tile coordinates, buffers and completion explicit. Remove dependence
on an unrelated persistent scheduler only when the resulting body still satisfies all protocols.

“CUTLASS-based” does not certify body quality after adaptation. Compare the original standalone
tactic, the adapted body, and the composed entry. Watch for layout conversion, mainloop/epilogue
storage overlap, register lifetime, warp-group participation and declared async completion points.
For staged bodies, a stage may return with a tracked outstanding operation; the enclosing protocol
must own its buffers/tokens until retirement. Atomic-tile bodies drain their declared accesses.

CUTLASS's [pipeline abstraction](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html)
separates producer acquisition/commit from consumer waiting/release. That is a useful local
storage contract, not a ready-made cross-CTA tensor dependency protocol. The MegaBake adapter
must connect local pipeline stages to its own tile publication and accumulator semantics.

Do not require MegaBake to reimplement every MMA instruction or library autotuner. A narrow adapter
with a few supported descriptors is a smaller and testable first backend.

## 5. Mirage MPK: useful device code, not a free generic FX frontend

Inspected commit: `17e9e36de583fe26a55be3fa0a6030f5c56a34d8`.

The [Hopper CUTLASS-based task](https://github.com/mirage-project/mirage/blob/17e9e36de583fe26a55be3fa0a6030f5c56a34d8/include/mirage/persistent_kernel/tasks/cute/hopper/gemm_ws_mpk.cuh)
contains a `CUTLASS_DEVICE` linear routine with templated dimensions, shared storage, warp-group
roles and MPK-specific TMA/pipeline dependencies. It is concrete evidence that good reusable math
can live below a standalone launch, but it is not a drop-in body for an arbitrary MegaBake CTA.

The [norm-linear task](https://github.com/mirage-project/mirage/blob/17e9e36de583fe26a55be3fa0a6030f5c56a34d8/include/mirage/persistent_kernel/tasks/hopper/norm_linear_hopper.cuh)
is another candidate reference for a transformer composite. Verify its dimensions, dtype,
normalization formula, scratch layout and state protocol before adapting it. Do not assume the
same implementation is legal or efficient on a different architecture or MIG configuration.

The inspected [persistent-kernel Python interface](https://github.com/mirage-project/mirage/blob/17e9e36de583fe26a55be3fa0a6030f5c56a34d8/python/mirage/mpk/persistent_kernel.py)
has explicit operation builders. The [MPK build path](https://github.com/mirage-project/mirage/blob/17e9e36de583fe26a55be3fa0a6030f5c56a34d8/python/mirage/mpk/mpk.py)
uses model builders. This path does not establish drop-in arbitrary-FX support. It also does not
prove that every other Mirage path lacks FX integration; the conclusion is deliberately scoped.

V3 owns the FX-to-semantics frontend and generated entry. Importing selected math with attribution
is consistent with that goal; silently invoking the complete MPK compiler/runtime and reporting
its result as MegaBake's generic compiler is not.

MPK's footprint/event concepts inform the primary V3 pipeline now. Do not import its entire queue
runtime just to reuse a GEMM. See the
[persistent runtime](https://github.com/mirage-project/mirage/blob/17e9e36de583fe26a55be3fa0a6030f5c56a34d8/include/mirage/persistent_kernel/persistent_kernel.cuh)
and [atomic protocols](https://github.com/mirage-project/mirage/blob/17e9e36de583fe26a55be3fa0a6030f5c56a34d8/include/mirage/persistent_kernel/mpk_atoms.cuh)
when designing scoped publication and tile readiness.

Source-level caution: the inspected runtime's `execute_worker` path prefetches task descriptors,
waits for a dependent event, invokes `_execute_task`, then publishes completion. That observation
alone does not demonstrate cross-task **weight** prefetch in this path. The paper describes a
richer pipelining mechanism; trace the exact implementation selected for reuse before attributing
that mechanism to it. Descriptor prefetch, intra-body K staging and inter-task tensor prefetch
are three different optimizations.

### Hazy's staged implementation: an additional reference

Inspected commit: `7309cec801537b61fea3b50d7dfe454a6cde578e` of HazyResearch/Megakernels.
Its [entry](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/include/megakernel.cuh)
dispatches separate controller, loader, launcher, consumer and storer roles. The
[matvec pipeline](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/demos/low-latency-llama/matvec_pipeline.cuh)
tracks weight arrival, weight retirement and output stages separately. These are concrete stage
interfaces, not just a list of operators placed in one entry.

The [down/output projection body](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/demos/low-latency-llama/matvec_adds.cu)
consumes reduction slices and uses additive stores. It distinguishes source-read wait from full
store completion before signaling. Its particular cast/add order and volatile polling are not
automatically V3's numerical or memory-order contract. V3 explicitly compares owner-held FP32
continuations with global partial/finalizer alternatives; it does not copy this body uncritically.

The [default configuration](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/include/config.cuh)
uses 16 consumer warps plus four other warps, or 640 threads. Its page and register configuration
is part of that implementation, not a 128/256-thread drop-in. The
[latency scheduler](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/megakernels/demos/latency/scheduler.py)
constructs model-shaped instruction assignments. It supplies a scheduling reference, not the
generic FX frontend MegaBake still needs. Reusing the entire runtime is not required to reuse
its separation of readiness, staging and retirement.

## 6. Luminal: preserve alternatives and distinguish execution levels

Inspected commit: `d18376d184172616ab3980309f524299a02595ef`.

Its [CUDA-lite README](https://github.com/luminal-ai/luminal/blob/d18376d184172616ab3980309f524299a02595ef/crates/luminal_cuda_lite/README.md)
distinguishes opaque host operations, launchable kernels and composable block operations. It also
states that legal alternatives should survive until selection. This is useful architectural
discipline: semantic equivalence does not imply identical runtime interfaces or performance.

The current [GEMV implementation](https://github.com/luminal-ai/luminal/blob/d18376d184172616ab3980309f524299a02595ef/crates/luminal_cuda_lite/src/kernel/gemv.rs)
provides a specialized M=1 mapping as a KernelOp. It is a mapping reference, not automatically an
in-grid device function. The inspected backend also contains explicit
[cuBLASLt host operations](https://github.com/luminal-ai/luminal/blob/d18376d184172616ab3980309f524299a02595ef/crates/luminal_cuda_lite/src/host/cublaslt/mod.rs)
and [CUDA Graph composition](https://github.com/luminal-ai/luminal/blob/d18376d184172616ab3980309f524299a02595ef/crates/luminal_cuda_lite/src/kernel/cuda_graph.rs).

The historical [megakernel article](https://blog.luminal.com/p/compiling-models-to-megakernels)
explains block-level composition and dependency scheduling. Do not assume every historical design
is currently exposed as a complete working BlockOp path in the inspected checkout. Revalidate the
specific implementation before copying an interface or quoting a performance result.

V3 borrows the useful separation, not a mandatory equality-saturation engine. A bounded candidate
set is enough to test the current hypothesis. More general search needs an observed limitation.

## 7. The MegaBake device-body contract

The proposed adapter must declare:

```text
semantics and numerical policy
supported target features and toolchain/library versions
shape/layout/alignment/tail guards
logical output tile and complete read/reduction footprint
required block shape and participating thread/warp roles
capabilities: atomic tile, separate preload, reduction continuation, early release
register-accumulator and shared-scratch lifetime requirements
host descriptors, addresses, workspace and setup lifetime
stage preconditions; issued/completed async operations; source retirement points
accumulator initialization/update/finalization and numerical-order contract
output ownership, publication scope, resource release and collective participation
```

It must not secretly launch child grids, allocate unaccounted storage, assume `blockIdx` is its
logical tile index, or leave **untracked** operations accessing scratch after a stage returns.
For an atomic tile, its declared accesses are drained at return. A staged body transfers explicit
operation/lifetime tokens to the enclosing generated program; those tokens cannot be discarded.
It cannot
require an incompatible number of threads just because the caller has extra idle warps; whether
extra threads can be gated is part of the body's synchronization contract.
No body may introduce an unadvertised grid collective inside a cohort-specific path. Global
joins belong to the verified plan and must be reached by every participating worker as required.

Each candidate is tested under the complete entry's chosen block shape and resource envelope.
The verifier may reject a legal standalone body because it is incompatible with other required
regions. That is a legitimate strict-backend limitation, not permission to misreport a split graph.
Registers held across a reduction continuation count against the entire generated entry. Giving
an API a continuation name does not make its accumulator cheap or its thread roles compatible.

## 8. Minimum reuse experiment

Use the hot shape list from the available real models. Build one lean K-parallel GEMV and one
target-supported tensor-core body. Include cuBLASDx when a compatible release/interface supports
the test; use selected MPK/CUTLASS code as an adaptation reference where useful.

For each shape, validate outputs, compare against the vendor operation with identical inputs,
record packing/setup, then embed the candidate in a representative persistent entry. Test two
adjacent tile loads/computations with and without lookahead, and the gate/down continuation
interface when claimed. Include live producer/consumer storage in the composition envelope.
Keep a body only if its combined cost helps the end-to-end budget. Add another tactic when a losing hot
shape warrants it. No four-library tournament is a prerequisite to this experiment.

Preserve source notices and record upstream commit/header versions for any future imported code.
The inspected MPK task files carry Apache-2.0 notices; check each selected file and bundled
dependency rather than assuming one repository-level label covers every asset. No upstream code
was copied into MegaBake by this documentation change.

## 9. Do not leave the non-linear path as an afterthought

Competitive linears are necessary for the current bottleneck, not sufficient for a decoder.
The first device portfolio also needs lean, semantically exact norm, activation, positional and
attention/state bodies. Prefer direct expressions/reductions over the current broad interpreted
pointwise machinery when the graph is specialized.

For ordinary batch-one decode, start with an online-softmax attention body that streams valid
K/V tiles without materializing the full score/probability matrices. The
[FlashAttention paper](https://arxiv.org/abs/2205.14135) supplies the IO-aware attention principle;
decode-specific tiling and composition remain V3 implementation work, not a claim that a prefill
kernel is the right decode kernel.

For one query, maintain running maximum `m`, normalizer `l` and weighted-value accumulator `a`.
For a nonempty valid score tile `s` and its values `V`, the real-arithmetic update is:

```text
m_new = max(m, max(s))
rescale = exp(m - m_new)
p = exp(s - m_new)
l_new = rescale * l + sum(p)
a_new = rescale * a + sum_j(p_j * V_j)
output = a / l after all valid tiles
```

This is an algorithm outline, not an approved floating-point lowering. Initialize the first valid
tile safely; handle empty/all-masked rows according to the reference instead of evaluating an
undefined `-inf - -inf` expression. Apply mask, scale, head mapping and cast/accumulation policy
exactly as the semantic contract requires.

Compare a head/query tile mapping with a split-context option only when context and parallelism
justify it. Split-context requires merging partial maxima, normalizers and accumulators with
correct rescaling, plus storage and synchronization. GQA can reuse K/V across query heads, but
holding too many heads' accumulators can increase registers and damage the full entry.

Cache write completion and valid-length/position metadata must precede the attention reads that
depend on them. A current-token cache update is not optional housekeeping. RoPE/cache fusion is
useful only when the same tile ownership and numerical contract allow it.

For RMSNorm and gating, compare materialized vector controls with local epilogues and chunked
gate/up fusion, preserving casts. Avoid duplicating a norm reduction for thousands of vocabulary
tiles by default. Attention publishes complete head regions into the primary pipelined schedule;
the MLP can consume complete activation chunks through a supported down-projection continuation.
Measure the non-linear path and its overlap in E3/E4 so a successful GEMV experiment cannot conceal
a new attention, state, reduction or pipeline-resource bottleneck.
