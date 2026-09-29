# MegaBake V3 architecture

Status: CUDA-first design revised 2026-09-27. This is the normative architecture for the V3 documents. It is a proposal, not an implemented or benchmarked V3 result. See the [index](MEGABAKE_V3_README.md), [historical evidence](MEGABAKE_V3_GPU_REANALYSIS.md) and [decision ledger](MEGABAKE_V3_RESEARCH_AND_DECISIONS.md).

## 1. Decision and scope

Build a compiler whose input is a complete Torch FX `GraphModule` or `ExportedProgram` plus explicit invocation, state, shape and numerical contracts. Its first target is one owned CUDA compute grid for a cached Hugging Face Transformers decode step. It must preserve logits, required state updates and output ownership, then demonstrate a complete-step win over the strongest equivalent `torch.compile` execution on the same device. An external-library or multi-grid fallback is a separately reported artifact.

The first workload matrix covers FP16/BF16, batch one and small batch, short and long valid context, and at least two structurally different decoder families. The first performance result may use one available GPU; claims across SM generations require measurements on those generations. Full-model capture, strict compilation and performance admission are separate gates.

The hard constraint is **quality of embedded math**. The best baseline may call a private cuBLAS/cuBLASLt tactic that cannot be invoked as a device function inside our grid. A generic correct contraction body does not inherit its performance. The compiler must create, adapt and select device-callable tactics competitive enough after composition. Historical V2 body deficits in the [GPU audit](MEGABAKE_V3_GPU_REANALYSIS.md) make this the first feasibility question.

One grid does not imply one CTA, one tile per CTA, no global intermediates, zero setup cost, or a kernel kept alive across token invocations. CUDA Graphs, vendor kernels and programmatic dependent launch can offer strong multi-grid controls. The strict research result remains one owned compute grid; the ordinary callable can select a disclosed fallback.

## 2. Compiler center: semantic tensor work rather than a FatOp cover

The compiler retains normalized FX as an executable reference. Its canonical analysis is an `IndexedTensorProgram` that attaches typed tensor computations, effects and region structure to FX origins. A supported operation describes iteration indices, input index maps, output maps, reduction/scan axes, masks, casts, aliasing and state transitions. Ordinary maps, broadcasts, contractions, reductions, views, simple indexing and functional state updates have a generic lowering path.

One logical state effect can be realized by a group of FX writers. For example, a fixed-capacity cache with a leading layer and key/value axis may lower to per-layer `index_copy` operations over proved `select` views, followed by `stack` operations that assemble the new cache. The effect proof must trace each old-state view to the declared input, prove every writer uses the same guarded append position, prove destination regions are disjoint or explicitly ordered, and prove the returned state assembles those writes while retaining untouched bytes. A single `source_id` may identify that aggregate result rather than one writer node. A frontend that cannot verify the whole group must reject strict lowering; it must not treat each member as an independent transition or silently ignore the effect.

Named recognizers such as `Linear`, `RMSNorm`, `RoPE`, `SDPA`, gated MLP and routed expert patterns are **guarded algorithm proposals**. Each records its reference expansion and exact legal preconditions. They can expose online softmax, special layouts, combined projections or streamed reductions. They never determine whether an otherwise supported primitive graph can be compiled. A new arrangement of existing primitive operations should require no model-family switch or new FatOp. A genuinely new custom operation or recurrence needs its own semantic/effect contract and may need a specialized algorithm to be fast.

This is a substantive replacement of the old FatOp-first semantic cover. It does not require importing all of MLIR, a universal hardware DSL, or an unrestricted e-graph. Begin with a small indexed-compute vocabulary and bounded rewrite rules. Keep an untransformed alternative until a guarded transformation is verified and selected. The [IR document](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) defines the exact records and pass order.

**Reuse PyTorch's compiler stack deliberately.** `torch.export` supplies the full graph signature and selectable decompositions; symbolic/fake-tensor machinery and selected Inductor FX passes can supply candidate facts and normalized pure subgraphs through a PyTorch-versioned adapter. A bounded experiment may translate Inductor's pure `Pointwise`/`Reduction` loop descriptions into our indexed operations if origin, guards, casts and effects survive. Inductor's autotune choices, generated kernels and fusion groups are also valuable exact-shape performance teachers. They do not become device-callable `BodyTacticSpec`s merely because Inductor can launch them: the owned-grid body still needs a compatible tile coordinate, participation, scratch and publication contract. [PyTorch 2.6 export](https://docs.pytorch.org/docs/2.6/export.html), [Inductor 2.6 loop IR](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/ir.py), [Inductor 2.6 compile flow](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/compile_fx.py). The [IR reuse policy](MEGABAKE_V3_IR_AND_REUSE_PLAN.md#2-capture-and-the-inductor-handoff) specifies the admission checks.

A verified `RepeatRegion` over the indexed program records iteration bounds, per-iteration parameter/state bindings, carried values, exceptional variants and reference meaning. The compiler may lower it to a device loop, partial unroll or flat schedule. `LayerSummary` remains useful analysis, but it is insufficient once cross-layer lifetime and prefetch decisions depend on explicit iteration. Model configuration and module names are hints; FX and state bindings establish meaning.

## 3. Algorithm choices precede physical scheduling

For each semantic computation, enumerate a bounded, equivalence-checked algorithm menu. The menu is not a single greedy fused cover:

| Work | Examples of legal alternatives to investigate |
|---|---|
| Low-batch projection | K-parallel SIMT GEMV; output-channel-major tensor-core GEMM; legal padding; smaller tiles or split-K; packed stable weights |
| Larger GEMM | Target-supported MMA mainloops, pipeline depths, layouts, epilogues and persistent tile scheduling |
| QKV | Separate projections; packed weight and combined contraction; consumer-aligned head-group tiles |
| Gate/up/down | Paired gate/up output and exact gating; materialized hidden vector; continued down reduction; explicit split partials |
| Norm | Compute once and materialize; local reuse; guarded fused pre/post processing; measured pure recomputation |
| Attention | Online-softmax decode; split-context with correct combine; target/cache-specific layouts |
| Repeated region | Flat specialization, template loop, partial unroll, different activation owners and weight prefetch depths |

For `Y = X Wᵀ` with `X[B,K]` and `W[N,K]`, the equivalent `Yᵀ = W Xᵀ` places output channels on the large GEMM axis. This can give a tensor-core body useful work for narrow B, subject to padding, occupancy, layout and numerical costs. A 64-channel tile yields only nine independent tiles for N=576 before splitting K; a K-parallel SIMT body may be better. On another shape or SM the tensor-core tactic may win. The compiler generates both where legal and measures rather than selecting by `B=1` or GPU product name. Mirage's [Hopper device body](https://github.com/mirage-project/mirage/blob/mpk/include/mirage/persistent_kernel/tasks/cute/hopper/gemm_ws_mpk.cuh) and Hazy's [different Hopper/B200 body choices](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) motivate this shape-and-target portfolio.

Algorithm, tile and layout alternatives form a bounded search. The body provider returns a Pareto set rather than one isolated winner: latency, global traffic, tile granularity, entry participation, registers, shared storage, code size, epilogue access and stage capabilities all matter. A fast full-K body can beat a streamed reduction despite later consumer readiness; a fused projection can improve matrix quality while delaying some heads. Keep these counterfactuals until whole-entry measurement.

## 4. Body providers and quality admission

A `BodyTacticSpec` has semantic/numerical guards, target/toolchain requirements, exact tile and full-reduction footprints, output ownership, acceptable enclosing participation configurations, data layouts, host descriptor/binding lifetime, scratch and accumulator lifetime, async completion rules, optional stage boundaries and source provenance. The generic body generator provides correctness coverage for supported indexed work. Specialized providers supply tuned SIMT and architecture-specific tensor-core implementations, including adapted CUTLASS/CuTe or cuBLASDx when compatible.

The competitive path is also a **body compiler**, not a catalog of checkpoint kernels. For each indexed contraction or reduction and each legal algorithm choice, a target provider generates a bounded schedule space: output/reduction tile shapes, lane/warp/CTA mapping, vectorization, MMA instruction family, copy/mainloop stages, accumulator placement, epilogue, split ownership and persistent tile iteration. Reject variants that violate guards or static resource limits; compile and measure a small Pareto frontier at exact shapes, first standalone and then in a lean owner entry. MPK's reference-to-device task-body search motivates this local superoptimization. CUDA primitive backends and reusable algorithm rules may still need engineering per new operation or SM generation, but a new model arrangement should not require a model-specific body file. [Body generation plan](MEGABAKE_V3_KERNEL_REUSE.md)

A body can be `ATOMIC_TILE`, `PRELOADABLE`, `STREAM_REDUCTION` and/or `EARLY_RELEASE` only when its actual device interface proves that capability. An internally pipelined GEMM does not automatically expose cross-operator weight preloading or a continued reduction. No body hides another compute grid, undocumented scratch, or an extra global synchronization.

The current cuBLASDx pipelined interface can stage global GEMM loads and compute, but creates pipeline metadata on the host, has tile-divisibility and depth constraints, and exposes a required launch block shape and a `__grid_constant__` device handle. Reusable accumulation and fixed block-size modes can change performance. It is a measured provider with explicit descriptor/setup costs, not the sole basis of the compiler. [cuBLASDx pipeline documentation](https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html) The [reuse document](MEGABAKE_V3_KERNEL_REUSE.md) specifies the adaptation experiments.

Inventory hot exact shapes from the actual FX step and benchmark the baseline's selected vendor paths. Compare body tactics as standalone kernels and in a minimal representative persistent entry. Preserve fused-body alternatives even when an isolated operation is slower, if a conservatively bounded end-to-end saving could pay for the deficit. For final admission, compile and measure the whole entry. Body quality is a feasibility gate and a search signal, not a proof based on summed isolated times.

## 5. Logical dataflow and target search

`LogicalExecutionPlan` contains parametric task domains, index/read/write maps, reduction contributors and finalizers, state/effect order, exact producer relations, movement opportunities, and typed readiness/release constraints. Logical tiles can greatly outnumber target workers. It contains no CTA count, warp roles, CUDA memory name, fixed lookahead depth, target cohort split or mandated grid barrier. A logical relation may require local forwarding, global materialization or a collective with an explicit reachability condition; the target either realizes it or rejects that candidate.

The backend participates in search rather than merely lowering a frozen logical plan. It proposes target-specific tile refinements, body tactics, physical layouts and storage, participant placement, static/dynamic schedules, transport and invocation shape. When such a choice changes tile footprints or dependency granularity, the logical plan is regenerated and reverified. `TargetExecutionPlan` records one resolved physical candidate with compiled-resource and runtime-admission states. [IR contracts](MEGABAKE_V3_IR_AND_REUSE_PLAN.md)

The initial CUDA scheduler is a low-overhead resident cooperative grid with statically generated worker programs for regular dense decode. A barrier schedule provides a simple control. Head-ready attention, gate/down chunk overlap and cross-task weight lookahead are optional measured variants. Dynamic or hybrid dispatch is a backend option for data-dependent expert work or measured imbalance, using the same logical dependency relations. Luminal's [symbolic block domains](https://blog.luminal.com/p/compiling-models-to-megakernels) and MPK's [task/event graph](https://arxiv.org/html/2512.22219v2) inform this representation; neither dictates our runtime policy.

## 6. Whole-region storage, synchronization and resources

Optimize over a verified repeated region when cross-layer decisions matter. Weights can have known addresses before their activations; this creates an early-movement opportunity, not an automatic profitable prefetch. Activation ownership and handoff are equally important. Inferact's [TPU megakernel](https://inferact.ai/blog/tpu-megakernels) demonstrates layer-ahead DMA under large VMEM and mesh-specific schedules. CUDA shared memory, registers, tensor memory and L2 have different capacity and reachability. Hazy measured substantial activation store/consistency/reload time even after weight stalls were reduced. [Hazy](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)

The physical planner distinguishes address known, source ready, copy issued, source access retired, destination visible, computation complete, reduction final, output published and storage reusable. It allocates from proven partial-order lifetimes rather than a predicted timing chart. Target-specific memory/proxy fences, barrier scopes, descriptor lifetimes and collective participation are checked. A data DAG without a worker/resource progress proof is insufficient. [Pipeline protocol](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md)

A CUDA entry has one compiled resource envelope: peak register allocation, shared storage, local-memory spills and instruction footprint affect all phases. Time-multiplexing can overlay mutually exclusive scratch, but does not make register allocation or launch residency phase-specific. Compile actual candidate mixtures and use occupancy/cooperative-launch APIs before admission. The selected block configuration must be compatible with every included body; this may rule out an otherwise fast cuBLASDx or CUTLASS tactic. [Hardware model](MEGABAKE_V3_HARDWARE_MODEL.md)

## 7. SM and toolchain portability

Semantic algorithms and logical dependencies are shared. Physical body implementations, supported instructions, pipelines and measured costs are target-qualified. The CUDA adapter queries the selected device and visible partition, compiles legal variants for supported feature sets, and tunes complete-entry configurations per workload bucket. It does not use numeric `SM >= n` as a substitute for feature legality or assume one H200 block size and shared-memory limit works elsewhere. NVIDIA architecture-specific `_a` features require an exact target, while family-specific `_f` targets have narrower compatibility than baseline code. [CUDA feature-set documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

Artifact keys include graph/reference semantics, guards, numerical policy, weight/layout preparation, SM feature target, visible resource profile where relevant, compiler/CUDA/provider versions, body mixture and binding ABI. Correctness-compatible cubins may be reused across a documented compatibility set; performance tuning remains target and workload specific. A new CUDA/toolchain version can change resource reports or library behavior and invalidates affected measurements.

The shared compiler keeps TPU open through indexed semantics, parametric work/dependence and repeated-stateful regions. A TPU backend would supply its own Pallas/Mosaic bodies, VMEM/SMEM/HBM allocation, DMA/semaphore contracts, TensorCore/mesh placement, collectives and runtime. It is not a translation of CUDA CTA programs, and portability is not claimed until another backend runs correctly and is measured.

## 8. Invocation, fallback and measurement

A session owns stable weights, optional packing, cache state, typed workspace and output-lifetime policy. The invocation validates guards, binds changing inputs and state, launches the selected entry, and returns the declared outputs. Copies, descriptor updates, event initialization, resets and output handling are charged to the relevant complete-call timing view. Repeated calls must advance or deliberately restore cache state under an explicit benchmark contract.

Compile two distinct plan classes from the same semantic program:

- `StrictGridPlan`: all required compute occurs in one owned CUDA grid; a measured win is counted only when this artifact passes correctness, resource admission and strongest-baseline comparison.
- `ExternalPlan`: may retain cuBLAS/cuBLASLt, other vendor kernels, CUDA Graph capture and legal cross-kernel overlap. It can be a faster product choice or fallback; it is reported separately.

The selector does not hide a strict loss behind the external plan. It reports captured, strict-compiled and performance-admitted coverage independently. The strongest equivalent `torch.compile` configuration gets matched weights, shape, cache, precision, input/output lifetime, setup accounting and legal tuning/capture. The [performance protocol](MEGABAKE_V3_PERFORMANCE_MODEL.md) governs claims.

## 9. First implementation and invariants

The first vertical path is: complete-step capture and matched baseline; exact-shape hot-body experiments; a thin indexed-FX-to-body-to-one-grid block; one correct full cached step; whole-region variants; a held-out family; then additional SMs and irregular workloads. The [implementation plan](MEGABAKE_V3_IMPLEMENTATION.md) gives reviewable work orders. Large generic schedulers, binary cuBLAS extraction, universal equality saturation and TPU implementation are outside this first path.

Invariants:

1. FX reference semantics, live outputs, state effects and numerical boundaries are retained through every rewrite.
2. No model name is needed to lower supported indexed primitives; named patterns have guards and reference expansions.
3. Algorithm, body, tile, layout, ownership and schedule choices remain alternatives until target feedback resolves them.
4. A logical task is not a CUDA CTA; physical placement and movement belong to a target plan.
5. Only verified device-callable work enters a strict grid. Vendor host calls remain external plans.
6. Target features, compiled resources, participation, storage lifetimes and forward progress are checked on the actual entry.
7. No unknown cost is silently zero; no fallback counts as a strict win; no isolated-body timing certifies a complete-step result.

## 10. Compiler handoff contract, stage by stage

An implementation agent should be able to hand one stage's result to the next without inventing a missing meaning. The order below establishes semantics first, then searches over physical choices; the search may revisit alternatives but may not weaken earlier guards.

| Stage | Input → required output | Hard rejection at this boundary |
|---|---|---|
| Capture | Complete FX/ExportedProgram + `WorkloadSpec`/numerical/state ABI → executable reference, bindings, output tree, effects, versions | Graph fragment, missing state output or unresolved lifted binding |
| Normalize/facts | Capture → copied version-pinned FX, origin map, proven shape/stride/alias/cast/effect facts | State changed by pass, guessed alias/alignment or dropped effect |
| Indexed semantics | FX/facts → maps, reductions, contractions, indexing, state transitions and local references | Live FX node left as opaque reference while claiming strict support |
| Algorithm choices | Indexed program → guarded equivalent alternatives plus unfused path | Pattern name without reference relation/numerical guard |
| Body generation | Indexed algorithm + target → correct generated body and bounded target-qualified tactics | Host-only/vendor launch relabelled device-callable |
| Logical work | Chosen body footprint → parametric tile domains, exact producer relations and lifetimes | Missing writer, duplicate effect, premature reduction finalization |
| Physical search | Logical work + body tactics + target → worker program, storage, events and progress argument | Incompatible CTA roles, unsafe wait or unsupported movement |
| Compile/admit | Complete selected entry → actual resource report and cooperative worker bound | Resource/feature/argument limit exceeded |
| Runtime/measure | Admitted entry + guarded session → logits/new state and complete-call evidence | Incorrect state/output, hidden second compute grid or unqualified baseline |

The body stage precedes the **final** logical tiling because body output/K footprints change dependence granularity. A provisional conservative tile plan can be built earlier to test generic lowering. Any body change that alters tile footprints regenerates and reverifies logical work. The backend does not merely lower a frozen device-neutral schedule; it proposes physical choices and can ask for a new legal tile refinement.

## 11. Current checkout and extension rule

The checked-in V3 code already has CPU `WorkloadSpec`/`NumericalPolicy` records, capture/normalization/facts, named semantic matchers, a composite enumerator, `LayerSummary` and tiny fixtures. Those are useful partial inputs. The current named `SemanticGraph` plus leftover `ReferenceRegion` is **not** the canonical indexed program and does not guarantee a strict body for unmatched FX. The legacy public `compile_fx` and HF wrapper are separate; the wrapper uses `use_cache=False`. [Implementation inventory](MEGABAKE_V3_IMPLEMENTATION.md#2-what-the-current-repository-gives-us)

A new arrangement of supported maps/reductions/contractions/views/indexing should traverse the generic body compiler without a model name. A new mathematical primitive needs a typed indexed/effect semantics and conservative body. A new fast implementation of an existing primitive belongs in the target body provider; a new SM instruction belongs in a target feature variant. A new HF checkpoint is a **validation input**, not a dispatch key. This rule makes the original FatOp maintenance concern measurable: the held-out family either compiles through known semantics, exposes a reusable primitive gap, or fails the stated breadth goal.

The initial product may select an `ExternalPlan` when private cuBLAS/cuBLASLt bodies are faster. The research claim remains a correct one-grid `StrictGridPlan` beating the strongest matched `torch.compile` on the predeclared cells. The [performance contract](MEGABAKE_V3_PERFORMANCE_MODEL.md#9-predeclared-scorecard-for-the-first-claim) defines the first matrix and validation rule. A correct strict loss is reported and guides the next body or physical-plan experiment; no document promises that one grid will win universally.
