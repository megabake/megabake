# MegaBake V3: competitive device bodies and reuse

Status: proposed CUDA body architecture, revised 2026-09-27. No V3 body or cuBLASDx adapter is benchmarked by this document. The [architecture](MEGABAKE_V3_ARCHITECTURE.md) owns the strict one-grid result; this file owns the quality and composability contract.

## 1. The central performance problem

The strongest `torch.compile` path may use cuBLAS/cuBLASLt kernels selected for an exact shape, dtype, layout and GPU. A strict owned grid cannot call those host APIs as device functions. Having a correct generic GEMM inside the grid is not enough: the historical V2 [shape evidence](MEGABAKE_V3_GPU_REANALYSIS.md) contains large vendor-relative body deficits. Launch savings or a clever scheduler cannot be assumed to repay them.

The architecture therefore has two body sources:

1. **Generated coverage bodies:** lower supported indexed maps, reductions, contractions, indexing and state work to device-callable CUDA. These establish correct strict compilation for unfamiliar arrangements of supported operations.
2. **Competitive tactic providers:** generate or adapt shape- and SM-specific SIMT, tensor-core and specialized attention bodies. These establish a plausible strict performance win for hot work.

A provider is reusable across model families because it is keyed by computation semantics and exact shape/layout, not a checkpoint name. A new mathematical primitive can require a new provider. Mirage MPK's [reference-to-device-task generation](https://arxiv.org/html/2512.22219v2) is a useful precedent; it is not evidence that a generic generated body automatically equals private cuBLAS performance.

## 2. Execution levels that must stay distinct

| Mechanism | Can supply strict in-grid math? | Role |
|---|---|---|
| cuBLAS/cuBLASLt host API | No | Best matched baseline or `ExternalPlan` |
| Captured private `CUfunction` | No device-callable interface follows from the handle | Diagnostic tactic identity, not code reuse |
| CUDA Graph or device graph launch | Constituent kernels remain separate grids | Vendor-preserving product/control path |
| CUDA dynamic parallelism | Launches child grids | Does not satisfy the strict one-grid contract |
| Inductor-generated Triton/CUDA kernel or autotune choice | Not as emitted; it is a launchable kernel candidate | Baseline/`ExternalPlan` and source of algorithm, layout and tile hypotheses |
| cuBLASDx block GEMM | Potentially, after caller supplies movement/tiling | Candidate tactic |
| cuBLASDx pipelined GEMM | Potentially, with host descriptor/entry adaptation | Candidate global-load-to-epilogue tactic |
| CUTLASS/CuTe C++ components | Yes after adapting participation, pipeline and pointer binding | Controllable tensor-core body source |
| Selected Mirage or Hazy device functions | Possibly after adaptation and source/contract review | Reference implementations or tactic sources |
| Generated SIMT indexed body | Yes | Coverage and sometimes fastest low-batch tactic |

A private library kernel's function/argument information does not reveal stable source or a callable device routine. Binary rewriting would need to reestablish indexing, barriers, async protocols and register allocation; it is a separate research project. GraCE's [vendor graph-node path](https://www.usenix.org/system/files/osdi26-ghosh.pdf) can improve dynamic graph binding while retaining a separate vendor launch. It cannot solve strict body composition.

Inductor should be an **active body teacher** for the exact hot shapes: retain which ATen, Triton or CUDA template won its legal autotune and, where inspectable, its tile/layout/padding and epilogue choices, generated source and register/shared usage. Record unknown details for opaque vendor kernels. Measure standalone and complete-step contributions. Seed the V3 tactic search from those choices, then measure the adapted tactic standalone, in a lean owner entry and in the complete mixed entry. The adaptation must expose arbitrary logical tile coordinates, compatible CTA roles, scratch/descriptor lifetime and output publication. Replaying an Inductor launch or treating its autotune cache timing as an in-grid cost does not establish that adaptation. [PyTorch `torch.compile` options](https://docs.pytorch.org/docs/stable/generated/torch.compile), [Inductor 2.6 loop/codegen path](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/compile_fx.py)

## 3. Body tactic contract

A selected `BodyTacticSpec` records:

```text
origin indexed computation + algorithm choice + exact numerical policy
supported SM feature set, CUDA/compiler/library versions and source provenance
shape/stride/layout/alignment/tail guards and any weight/activation preparation
output tile, full read/reduction footprint, writer and epilogue ownership
acceptable enclosing block/cluster configuration and participant roles
local/global/tensor-memory scratch, accumulator and descriptor lifetimes
atomic tile or separately preloadable/continued-reduction/early-release stages
async issue/completion, source retirement, output visibility and publication scope
measured standalone cost, lean-entry cost, compiled resource report and uncertainty
```

Every physical interface must allow a resident worker to execute logical tile coordinates chosen by the scheduler. It may not assume `blockIdx` is the semantic tile or that only one tile is ever executed per CTA. A body requiring a specific block dimension or collective participation may be incompatible with an otherwise attractive whole entry; report that explicitly. An `ATOMIC_TILE` drains its declared async accesses at return. A staged body transfers typed outstanding-operation and buffer-lifetime tokens to the enclosing program.

Keep a Pareto set across latency, register/shared footprint, supported enclosing block sizes, tile count, code size, fusion/epilogue access and stage capability. The fastest isolated body is not automatically the best composed body. Compile and inspect real mixtures before discarding a lighter tactic.

### Generated tactic search, not a model-kernel registry

The ordinary indexed body route generates a correct device implementation for every declared supported primitive. For hot structured work, each target provider also emits **schedule variants** from that same reference: output/reduction tile sizes; lanes, warps and CTA roles; vector width; serial versus parallel K; legal MMA primitive and operand layout; global-to-shared movement; mainloop stage count; accumulator location; epilogue and persistent tile iterator. A legal variant has an exact semantic/numerical guard and an enclosing-entry participation contract. This is a constrained task-body search inspired by [MPK's body superoptimization](https://arxiv.org/html/2512.22219v2), not unrestricted source synthesis or a promise that the resulting tactic equals private cuBLAS.

Search in two levels to control compilation cost. A shape/target filter removes impossible variants using tile counts, padded work, alignment, feature legality and estimated resource bounds; local microbenchmarks then keep several latency/resource/stage Pareto candidates. A lean owner-entry benchmark re-ranks them under common block shape and live scratch. Whole-entry search can request a different tile or a lower-resource variant, which regenerates exact dependence maps. Cache compiled variants by indexed semantics, shape/stride/numerical guard, target feature set and toolchain/provider version, never by model name. A provider can implement its schedule family through generated CUDA, an adapted CUTLASS collective or compatible cuBLASDx pipeline; a family of SM instructions is reusable across many FX models.

## 4. Low-batch contraction algorithms

Given `Y[B,N] = X[B,K] W[N,K]^T`, generate legal alternatives rather than a universal GEMV implementation:

| Tactic | Likely use | Cost or guard |
|---|---|---|
| K-parallel SIMT GEMV | Batch one, low output-channel count, architecture where tensor-core setup is costly | Lanes/warps reduce K; tune rows per CTA, vector width, warps per row and reduction scope |
| Output-channel-major tensor-core | Moderate/large N with BF16/FP16 and legal layout | Compute `Y^T = W X^T`; narrow batch axis may need padding; count output tiles and wasted MMA |
| Split-K or smaller output tiles | Too few output tiles to use the visible device | Partial traffic, finalizer, numerical order and synchronization cost |
| Conventional tiled GEMM | Larger batch or high reuse | Target-specific mainloop, pipeline depth, layout, epilogue and scheduler |
| Stable weight pack / QKV or gate-up pack | Improves access/tile shape across many invocations | Session setup and memory footprint; exact output/cast semantics |

The historical skinny V2 body assigned an output element to a thread and reduced K serially; the [GPU audit](MEGABAKE_V3_GPU_REANALYSIS.md) explains why this left too little useful parallelism for some shapes. The first SIMT body should give lanes K work and use a complete FP32 reduction before the declared cast/epilogue. A body must include real global loads, reduction and stores; timing only an MMA instruction is insufficient.

For narrow B, `Y^T = W X^T` makes N the large output axis while preserving K-contiguous weight rows. Mirage's [Hopper device body](https://github.com/mirage-project/mirage/blob/mpk/include/mirage/persistent_kernel/tasks/cute/hopper/gemm_ws_mpk.cuh) includes an operand-swapped, small-batch tensor-core design. Its exact tiling, warp roles and TMA setup are Hopper-specific, so it is a tactic reference, not a generic copied kernel. At N=576, 64-channel output tiles yield nine independent tensor-core tasks before K splitting; at N=4096 they yield 64. Those counts explain why different tactics may win on the same GPU. Padding and inactive output stores are charged and checked.

A split-K tactic may increase parallelism but changes reduction order and adds a combine. Its legality follows the numerical policy. For a streamed gate/down composite, an output-owner continuation is a distinct tactic: it must keep a legal accumulator and expose updates over completed hidden chunks. Do not present a full-K cuBLASDx GEMM as a continued reduction merely because its internal K loop is pipelined.

## 5. cuBLASDx: useful but constrained

The regular block GEMM interface assumes input tiles are provided in shared memory and leaves global tiling/loading/synchronization to the caller. The [pipelined interface](https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html) can start from global memory and internally stage asynchronous loads and MMA, including architecture-supported TMA/WGMMA/UTCMMA paths. It is the relevant candidate for full-projection body quality; regular `execute` alone does not reproduce a vendor global GEMM.

The pipelined API currently imposes important integration requirements:

- A host pipeline binds global tensors and creates metadata, including TMA descriptors when used. The device handle is passed by value as a `__grid_constant__` **global-kernel argument**, not a free device-function argument.
- Its default optimized path expects `get_block_dim()`; `fixed_blocksize` allows another choice but disables warp-specialized paths. Different body types in one grid may demand incompatible dimensions.
- Global dimensions must be divisible by descriptor tiles and depth must not exceed K tiles. Padding or a different tactic is necessary otherwise.
- A tile pipeline's destructive constructor/destructor occurs once per kernel execution; persistent loops use `reset_tile` after completing the prior tile's async work.
- User-owned reusable accumulation can disable register trading and reduce performance on some architectures. Its `finish_accumulation()` has a participation contract.

These are documented by NVIDIA's [pipeline guide](https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html). Multiple same-shape layers might be packed into a batched descriptor and selected by a batch coordinate, as NVIDIA's [batched pipeline example](https://github.com/NVIDIA/CUDALibrarySamples/blob/main/MathDx/cuBLASDx/19_gemm_batched/batched_gemm_pipeline.cu) illustrates. Heterogeneous layer shapes, many bindings and descriptor lifetimes still require a compiled compatibility experiment. No opaque handle is assumed to support arbitrary device-side pointer rebinding.

The current [cuBLASDx requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html) require CUDA Toolkit 13.0+, C++17 and a recent CUTLASS. Older CUDA builds need another provider or an explicitly pinned compatible release. Toolchain migration is a separate measured choice; cuBLASDx is optional in the provider registry.

## 6. CUTLASS/CuTe and target-qualified math

CUTLASS separates device adapters, kernel schedulers and lower collective/mainloop components. A target body may adapt a supported collective or a proven tile mainloop below its host launcher, preserving global load, pipeline, MMA, epilogue and synchronization behavior. The [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) describes the collective interface and participant roles. A C++ function marked device-callable still needs legal CTA roles, storage and async completion inside the selected entry.

The CUDA provider registry keys tactics by features and toolchain rather than a numeric `SM >=` rule. An Ampere/Ada warp MMA body, a Hopper TMA/WGMMA body, a data-center Blackwell tcgen05/TMEM body and an `sm_120` body need separate legality and performance evidence. A portable SIMT implementation supplies a correctness route when accelerated bodies are unavailable. Architecture-specific `_a` code is exact-target code; family-specific `_f` code has defined family compatibility. [CUDA compute-capability guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

Do not assume that CuTe DSL source is directly callable as a CUDA C++ `__device__` body. Adapt the appropriate source/component or generate the entire entry in a compatible toolchain. Preserve notices and pin versions for imported code. The target provider owns compilation, artifact keys and source compatibility, not the common indexed program.

## 7. Nonlinear and unfamiliar work

Generic scalar/indexed lowering supplies correct pointwise, norm-like and simple reduction work. Hot attention requires algorithm choices: online softmax over valid KV tiles, optional split-context reduction with correct max/normalizer rescaling, exact mask/head/cache semantics and target-specific layout. A model with paged KV, MLA, Gated DeltaNet, MoE routing or recurrent state may need new generic indexed primitives and specialist providers. This is operation-level extension rather than checkpoint-specific kernel assignment.

Gated MLP fusion must preserve actual activation and casts. For QKV, producer tiles can publish complete head groups only when their K reduction and required RoPE/cache work are finished. The selected body must expose that granularity. A large fused GEMM can be faster than head-ready scheduling; both remain candidates. The [pipeline document](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) specifies dependencies without forcing one tactic.

## 8. Admission experiment and rejection rules

For each hot shape, record baseline vendor operation, exact dtype/strides/numerical mode, working-set/cache condition, body tactic source, padding/packing, standalone latency, lean-entry latency, registers/shared/local memory and complete-step call count. Compare at least a K-parallel SIMT path and one supported tensor-core path where legal; use cuBLASDx and CUTLASS only when their source/toolchain contracts fit. Measure a minimal mixed entry with realistic live storage before whole-model scheduling work grows.

A large isolated deficit is a warning, not an automatic veto when a fused candidate removes measurable work. Bound the plausible launch/handoff/overlap savings conservatively, preserve mechanism-distinct alternatives, then measure complete candidates. If every legal embedded tactic for a dominant shape exceeds that budget, record a strict performance miss for the cell and improve the algorithm/body before investing in more scheduling. Keep a vendor-preserving `ExternalPlan` as a separate control and product path. No kernel source in this document has been imported or validated in MegaBake V3.

## 9. Implementable body ABI and collective rules

A body provider returns source plus metadata; the physical planner never calls an opaque “kernel” by name. The first CUDA ABI can be expressed conceptually as:

```text
BodyTacticSpec {
  indexed_origin_hash, algorithm_id, numerical_and_layout_guards,
  target_feature_and_toolchain_guard,
  logical_output_tile_map, reduction_footprint, active_tail_predicate,
  entry_block_shape_set, participating_warps_and_uniformity,
  host_binding_requirements, entry_argument_bytes,
  shared/register/tensor_memory_scratch_and_lifetimes,
  async_issue_wait_source_retirement_and_visibility,
  epilogue_and_output_publication_owner,
  stages: atomic_tile | explicit_preload | reduction_continuation | early_release,
  source_hash, compiled_resource_report, local_measurements
}
execute(entry_bindings, logical_tile_coordinate, worker_context, scratch_lease)
```

`execute` is an in-grid operation of the **selected owner entry**, not a child launch. An atomic body has no outstanding writes or source reads when it returns. A staged body can return only typed outstanding tokens that the worker program must later discharge before publication or scratch reuse. Body metadata must expose whether a collective requires the whole CTA, a warp group or a cluster; a worker cannot invoke such a body from a divergent branch with missing participants. A provider that assumes `blockIdx` is its output tile must be adapted to a logical tile coordinate or rejected for persistent reuse.

The entry may need multiple pipeline handles. For the cuBLASDx pipelined route, the host creates descriptors, passes handles as `__grid_constant__` **entry arguments** and keeps them alive for the call; the device-side tile pipeline is derived and reset under the documented lifecycle. A free-standing device function cannot receive the handle as an independently launched argument. Measure the number/size of handles, kernel-parameter budget, descriptor setup and pointer changes across token calls. Do not assume a descriptor bound to one layer can be rebound to another on device. [NVIDIA pipeline guide](https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html)

## 10. Bounded synthesis for a new indexed contraction

Given `Y[B,N] = X[B,K]W[N,K]^T`, the generic route first emits a correct full-reduction SIMT body. The competitive route enumerates a **target-qualified** grid of choices, pruned before compilation:

| Decision | Candidate families | Reject/prune when |
|---|---|---|
| Work shape | output N tile, B tile, optional K split | Too few output tiles for visible workers, excessive padded work or invalid numerical split |
| Thread mapping | lanes across K, warps per row, rows per CTA | Inactive lanes dominate, unsupported reduction scope or incompatible entry block size |
| Loads | scalar/vector weight access, activation broadcast, optional stable weight pack | Alignment/stride guard absent, pack cost cannot amortize |
| Tensor math | target legal warp MMA, WGMMA, tcgen05 or SIMT | Feature/toolchain unsupported, narrow padded B or descriptor incompatible |
| Pipeline | stage depth, async movement, shared/tensor-memory allocation | Resource cap, illegal async wait or lower occupancy than budget permits |
| Epilogue | separate materialization, bias/cast/fused map | FX live boundary or declared cast/effect would be lost |
| Persistent iteration | one tile versus repeated logical tiles per worker | Body cannot reset internal pipeline or release scratch correctly |

A local reference evaluator and numerical policy reject wrong candidates. Compile a bounded subset, measure exact shapes in standalone and lean-entry placements, and keep a Pareto frontier over latency, resources, tile granularity, block compatibility and available stages. The whole-entry search may choose a tactic that was second-best standalone. The next candidate is chosen from measured bottleneck data: K parallelism, output tile underfill, global movement, MMA utilization, registers/spills or entry coordination. No task card authorizes a blind Cartesian search over every tile/warp/stage combination.

A practical first search budget is at most 128 statically legal schedule tuples per hot shape, 32 compiled local candidates, 8 standalone finalists and 4 lean-entry finalists, retaining at least the best SIMT, best legal MMA and smallest-resource viable tactic when they differ. These numbers bound the **first pass**, not the quality ceiling: record why each candidate was pruned and expand the relevant dimension if all finalists miss the vendor budget or a resource cliff appears. Never prune the only correctness body. Whole-entry finalists are selected by actual compiled resources and fresh complete-call measurement.

For a new **model arrangement** of known indexed operations, the generator and providers should work without new model code. For a new **primitive** (for example a recurrence or paged gather), add its exact semantic/effect contract and a conservative body before a specialist provider. For a new **SM feature**, add a target provider variant and retune; do not edit model matchers. These are distinct extension points with distinct evidence.

## 11. Toolchain decision for the current checkout

`requirements.txt` currently pins PyTorch `2.14.0+cu130` and CUTLASS `3.8.0.0`. The PyTorch CUDA runtime does not select the host nvcc: record both independently. NVIDIA's currently documented cuBLASDx requires CUDA Toolkit 13.0+ and CUTLASS 4.4.1+, so V3R-008 is an **isolated provider experiment**, not an implicit upgrade of the main repository. [NVIDIA requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html) Record the actual selected nvcc, MathDx/CUTLASS headers, driver and target. A CUTLASS-only tactic compatible with the selected compiler can proceed independently; a cuBLASDx result from CUDA13 cannot be credited to a CUDA12 build.

If cuBLASDx is fast in isolation but its `get_block_dim()`, host handle or persistent tile lifecycle prevents composition, document that incompatibility and keep its result as a body experiment. If fixed block size or reusable accumulation makes composition legal but loses warp specialization/register trading, report both resource and latency effects. The fallback remains a vendor-preserving `ExternalPlan`; it does not make the strict body-quality gate pass. [Pipeline behavior](https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html)

## 12. Required admission record for every hot tactic

A hot tactic record contains the exact FX/indexed origin, algorithm, B/N/K, strides/alignment, dtype/casts, target feature set, toolchain/provider versions, body source hash, standalone and lean-entry raw timings, vendor operation and raw timing under the same cache condition, compiled registers/shared/local/spills/code, block/grid/tile counts, descriptor/pack setup, correctness cases and reason for retaining/rejecting it. The [performance model](MEGABAKE_V3_PERFORMANCE_MODEL.md) supplies the conservative whole-step budget. A summary such as “tensor cores are faster” or “cuBLASDx compiles” is not an admission record.
