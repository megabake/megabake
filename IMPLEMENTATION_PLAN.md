# MegaBake implementation plan

**Status: proposed work, revised 2026-10-05.** This document describes work to implement; its checkboxes do not imply that code or tests already exist.

[ARCHITECTURE.md](ARCHITECTURE.md) defines the design. This plan tells you what to build, in which order, how to check it, and which evidence to retain.

**The destination is the complete supported inference workload executing correctly in one optimized CUDA megakernel.** All stages serve that destination. A library launcher, eager fallback, separate per-block kernels or a silently shortened workload cannot count as successful MegaBake compilation.

## How to use this plan

1. Work through M0–M6 in order. Complete the correctness and one-launch gates before building on a stage. M7 adds explicit workload regimes and measured optimizations.
2. Within a stage, finish one vertical slice: input graph → representation → execution → check. Keep changes independently reviewable.
3. Create only the files needed now. Suggested names are locations for future implementation, not a request to scaffold them all.
4. Keep three results separate: correctness, one-kernel execution, and measured performance. A correct slow kernel can support the next development stage while its performance gap remains tracked. It does not fulfill the optimization objective by itself.
5. Save the exact workload, commands and artifacts. A passing GEMM fixture proves that fixture; full-model support requires the full-model gate.
6. Measure on an otherwise idle GPU after correctness passes. Separate compilation, profiling and steady-state timing.

The repository currently has `src/eager/verify_smollm_fx.py`, the wrapper `src/transformers/dump_smollm_fx.py`, an empty `src/megabake/__init__.py`, and saved trace artifacts. There is no implemented compiler pipeline or tracked compiler test suite to treat as complete.

## Stage map

| Stage | Build | Evidence required to move on |
| --- | --- | --- |
| M0 | Reliable source handoff | Same post-grad phase, valid runtime arguments, guards and outputs |
| M1 | Baselines and actual CuTe execution | Reproducible measurements and a real compiled CuTe kernel |
| M2 | Generic IR and GEMM/epilogue compilation | Complete fixture reaches one correct launch through the compiler |
| M3 | Cross-CTA handoff and multiple bodies | Visibility, progress, repeatability and body-layout compatibility |
| M4 | Task/event compiler and persistent MLP | Correct events, static workers, memory lifetimes and whole-MLP execution |
| M5 | Semantic definitions and full block | Complete block with norm, attention, RoPE, MLP and residuals |
| M6 | Full SmolLM megakernel | Complete model invocation, full outputs/state and one workload launch |
| M7 | Decode and continued optimization | Correct KV state; measured scheduling, storage and pipeline improvements |

The first major runtime proof is M3. Do not postpone it until every semantic operation has a matcher. Conversely, keep M0 and M1: a sophisticated scheduler is useless if the graph or performance comparison is wrong.

## M0 — Reliable source handoff

**Purpose:** know exactly which computation MegaBake receives and how its callable fits back into PyTorch.

**Start from:** the current capture script and the implementation inside the installed PyTorch wheel.

**Likely files:** `src/megabake/frontend.py`, the existing capture/CLI files, package constraints, and focused frontend tests.

### Build it

- [ ] **Pin the private integration.** Begin with the repository's tested PyTorch `2.14.0` setup. Align package constraints, check the version before private imports and record `torch.version.git_version`. An adjacent PyTorch checkout is not proof of the wheel's behavior.
- [ ] **Extract a reusable adapter.** Expose a backend for `torch.compile`. Investigate the pinned `torch._inductor.compile_fx.compile_fx` inner compiler callback and reuse its AOT preparation/runtime wrappers. Keep all private API handling in this adapter.
- [ ] **Make capture independent of executable caches.** Remove the capture script's `FxGraphCache` early-return path. During live handoff preparation, prevent AOT/Inductor executable-cache reuse from skipping the callback; validate the pinned release's `torch.compiler.config.force_disable_caches` behavior. Dynamo may still reuse a completed callable under valid guards.
- [ ] **Produce a consistent graph phase.** Run the selected view normalization, fake-tensor propagation and post-grad passes in their required context. Preserve output metadata, refresh tensor metadata as needed, lint the graph and recompile FX code after edits.
- [ ] **Preserve argument/output adaptation.** Pass the live graph and matching example inputs to the lowering callback. Return its callable through the AOT/Dynamo wrapper with the expected boxing convention. Do not rebuild runtime argument order from export placeholders.
- [ ] **Add an explicitly named reference executor.** Use it only to validate the handoff before CUDA lowering exists. Its result reports reference execution and no generated MegaBake kernel.
- [ ] **Declare initial support.** Inference, `torch.no_grad()`, `fullgraph=True`, concrete guarded shapes. Unsupported training or graph breaks fail with a clear reason.
- [ ] **Make capture reproducible.** The CLI records model revision, seed, batch/sequence sizes, dtype, attention implementation, output scope and tolerances. Write a new artifact directory for each run so an old success cannot survive a failed run.

The adapter's core boundary should be understandable as:

```text
PyTorch wrapper -> live post-grad graph + matching example inputs
                -> MegaBake lowering callback
                -> callable with that graph's runtime contract
                -> PyTorch wrapper returned to the caller
```

Example inputs guide specialization. Repeated execution must bind new runtime inputs and current weights. If export is retained for inspection, preserve its graph signature and constraints. [PyTorch backend contract](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html)

### Prove it

Build a tiny locally constructed transformer fixture so frontend tests do not require a downloaded model. Exercise CPU and CUDA where relevant:

- Instrument actual pass/callback execution with caches enabled and disabled; compare graph phase and normalized targets, not only output values.
- Change input contents, parameter contents, shapes and strides. Confirm valid reuse/recompilation or an explicit rejection.
- Check nested/duplicate outputs, alias relationships, input mutations and buffer updates through the wrappers.
- Check supported masks and multiple batch elements against eager execution.
- Reject unsupported PyTorch versions and training requests before entering unsupported private paths.

**Exit:** the handoff and wrapped reference callable satisfy their contracts. No megakernel claim is made yet.

**Save:** graph-phase dumps, source IDs, operator inventory, bindings, wrapper/effect notes, environment provenance and test commands/results. Record any wrapper work that could later generate an extra GPU launch.

## M1 — Baselines and actual CuTe execution

**Purpose:** establish a trustworthy speed target and prove the chosen backend can compile on the real GPU.

**Depends on:** M0 for compiler comparisons. Standalone toolchain inspection can happen earlier.

**Likely files:** one benchmark runner under `benchmarks/` and a small CuTe smoke test.

### Build it

- [ ] **Record the actual target.** Begin with the available H200 MIG / SM90 configuration. Query SM count, memory, device capability and MIG resources rather than copying a full-H200 configuration.
- [ ] **Define projection fixtures.** Start with `X[M,K] @ W[N,K].T`, BF16, `K=576`, `N=1536`, and both `M=4` and `M=256`. Record exact strides, seeds and output scope. Add an odd-sized case when testing supported tails.
- [ ] **Define cast-correct references.** Measure standalone GEMM and BF16 GEMM result → FP32 SiLU → BF16 output. Any bias follows the source expression's broadcast and rounding order.
- [ ] **Measure effective baselines.** Include eager/library GEMM, eager CUDA Graph replay, default Inductor, and suitable autotuned/CUDA Graph configurations. Record unsupported configurations and whether requested template autotuning actually ran on this MIG target.
- [ ] **Separate timing categories.** Record setup/first-call cost, warmup, repeated GPU-event measurements and synchronized host-call measurements. Keep raw samples and a clear summary such as median and spread. Profile separately for launch counts.
- [ ] **Compile and run CuTe.** Use the pinned DSL to build a small copy or pointwise kernel. Check an odd tail, changed inputs and a non-default PyTorch stream. Record which compiler packages the path actually uses; inspect `nvcc` only if that route uses it.

Release temporary benchmark outputs when they would otherwise change memory pressure. Match output scope and numerical policy across baselines. A four-token no-cache fixture is a projection/forward experiment, not a KV-cache decode benchmark.

**Exit:** repeatable baseline records for both M regimes and a real passing CuTe execution test.

**Save:** workload/reference definitions, raw timings, effective tuning configuration, launch traces, numerical errors and toolchain metadata.

## M2 — Minimal Compute IR and GEMM plus epilogue

**Purpose:** get the entire compiler path working for one small declared workload.

**Depends on:** M0–M1.

**Likely files:** small `ir.py`, `import_fx.py`, `verify.py`, `plan.py`, `kernels/gemm.py` and `lower_cute.py` modules under `src/megabake/`. Combine helpers until a real responsibility requires separation.

### M2.1 — Import the computation

- [ ] Define graph/value/op records, typed operands, concrete index maps and lightweight regions from the architecture. Keep GPU tiling and allocation outside Compute IR.
- [ ] Import the fixture's actual post-grad overloads using a target-to-importer table. Do not parse trace text or match node comments.
- [ ] Support the GEMM, pointwise, view/broadcast and cast forms this fixture needs. Inspect whether SiLU arrives decomposed and preserve its actual expression.
- [ ] Normalize transpose/view plus GEMM while retaining required copies and observable aliases. Preserve every live secondary output.
- [ ] Retain unsupported operations as source-backed diagnostics that block executable emission.
- [ ] Implement graph/region verification and a small reference evaluator for this supported IR. Compare imported and normalized IR against the source FX computation.

**Checks:** transposed weights, row/column broadcasts, scalar dtypes, BF16 rounding, multiple users, invalid use order, missing outputs and one unsupported overload. Each failure should identify its source node or violated invariant.

### M2.2 — Establish the reusable device body

- [ ] Reuse or implement a GEMM device body supported by the selected CuTe/toolchain/SM90 target. Confirm that its work can run inside our kernel; a library's host launcher does not qualify.
- [ ] Declare supported shapes/strides/dtypes, thread participation, input/output fragment layouts, shared-memory size/alignment, barrier usage, async operations and completion conditions.
- [ ] Validate standalone GEMM first. Include the stored transposed-weight layout and advertised tails. Record compiled registers, shared memory, spills and both M-regime timings.
- [ ] Attach the epilogue locally. Preserve the BF16 result rounding before FP32 activation without requiring an HBM store.
- [ ] Make layout transfer explicit. Direct register reuse requires compatible fragment ownership; otherwise describe and measure the conversion or shared-memory exchange.

Start with a run-to-completion body. It completes outstanding asynchronous work before returning. Do not add a generic body plugin system or prefetch API yet.

### M2.3 — Connect capture to one launch

- [ ] Choose a region covering the complete fixture and build its `DeviceTaskPlan` and one `KernelPlan` with runtime bindings. The task describes one output tile and the launch covers all such tiles.
- [ ] Verify coverage, maps, required reductions, body capabilities, ownership, storage and local barriers.
- [ ] Generate CuTe execution from that plan. Missing scheduling decisions are compiler errors rather than backend guesses.
- [ ] Connect the lowering callback to M0. The compiled path must never quietly return the reference executor.
- [ ] Enumerate a few supported body/tile variants. Compile, verify and time the full GEMM-plus-epilogue fixture; retain failed candidates and reasons.

**Exit:** a live captured graph passes import, verification, planning and lowering, then executes as one correct kernel. Repeat with new inputs/weights and the supported stream behavior. Use a warmed profiler trace to count workload launches; report compilation/tuning trials separately.

**Save:** source-linked FX/IR/plan dumps, body contract, generated code/compile diagnostics, full-kernel resource data, numerical checks and baseline comparison. Keep any remaining GEMM throughput gap visible.

## M3 — Prove cross-CTA handoff and multiple-body composition

**Purpose:** resolve synchronization and composition risks while the graph is still small enough to inspect by hand.

**Depends on:** M2.

**Likely files:** extend the plan/verifier/lowering modules and add minimal device-runtime code under `kernels/`. A general dynamic scheduler is unnecessary here.

### M3.1 — Establish the launch and initialization mechanism

- [ ] Make a fixture in which one CTA produces a transformed tile and a different CTA consumes it. Write down which CTA owns each write/read and the expected result.
- [ ] Check cooperative-launch support on the actual target and whether the CuTe launch path exposes it. Calculate the maximum legal resident grid from the complete compiled kernel and its shared-memory allocation.
- [ ] Use a supported cooperative launch for the first proof. If the launch API needs a small adapter, implement and validate it here. If the target cannot support this route, document and test a different forward-progress/initialization protocol before proceeding; reducing the grid size alone is not proof.
- [ ] Initialize invocation counters and required workspace state inside the same kernel, then use the supported grid synchronization before any worker observes them. Keep every required CTA in that collective phase, including otherwise idle CTAs.

A prototype that only works because a separate `zeros` or reset kernel ran first fails the one-launch gate. [CUDA cooperative-launch constraints](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION.html)

### M3.2 — Establish publication, readiness and reuse

- [ ] Finish producer writes and any asynchronous transfers before readiness publication. Coordinate all participating producer threads before the publishing thread signals.
- [ ] Implement a device-scope publication/observation protocol with the required release/acquire operations and instruction-specific fences. Document which operation establishes each happens-before edge.
- [ ] Begin with ordinary global-memory writes, a producer CTA synchronization, one elected thread's acquire-release counter increment, and a consumer leader's acquire polling followed by CTA synchronization. Validate the full publication chain and counter bounds. Extend this protocol to async/TMA bodies only after adding their required completion/fence operations.
- [ ] Extend the fixture to two producers and one consumer. Prove that observing the final count makes both producers' data visible. Do not assume an arbitrary atomic increment supplies this guarantee.
- [ ] Extend it to one producer with two consumers. Keep its buffer alive until both have finished reading.
- [ ] Use invocation-private state. Repeat with the same workspace after completion and with independent workspaces for independent invocations. Define the supported stream/concurrency contract.
- [ ] Write the progress argument: producer work can be scheduled while consumers wait, all bodies finish, and collective participants cannot exit early.

The CUDA memory model is the implementation reference; a high-level event diagram does not specify the required device instructions. [CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)

**Checks:** deliberately delayed producers, uneven work, fan-in, fan-out, empty worker lists, tails, poisoned/uninitialized workspace and repeated calls. Run potentially hanging tests in a subprocess with a bounded timeout. A timeout is a failure.

### M3.3 — Compose gate/up and their consumer

- [ ] Import two projections followed by cast-aware SiLU/multiply using generic IR.
- [ ] First try matching output tiles in one worker: gate GEMM, up GEMM, then the consumer. Retain all live outputs required by the fixture.
- [ ] Validate fragment compatibility. Compare direct reuse, a required conversion and any legal shared/global placement alternative.
- [ ] Measure the complete compiled kernel. Paired accumulators can raise register pressure; try smaller tiles or sequential bodies when necessary.
- [ ] Keep this fixture's scope explicit. It excludes down-projection; only M4's complete-MLP fixture will prove that additional dependency chain.

**Exit:** both the cross-CTA handoff fixture and the gate/up fixture pass numerical, visibility, lifetime, progress and one-launch checks.

**Save:** tile/event diagrams, launch-support evidence, compiled residency calculation, publication protocol, progress argument, repeated-call results and synchronization/resource timings.

## M4 — Compile tasks/events and execute a complete persistent MLP

**Purpose:** turn the M3 proof into the reusable runtime needed by the whole model.

**Depends on:** M3. Preserve its tested visibility and initialization protocol.

**Likely files:** the existing plan/verifier modules, a task-graph/scheduling module, the device runtime and a small tuning helper if several fixtures share it.

### M4.1 — Build the task graph from tile accesses

- [ ] Expand selected body/tiling choices into task instances. Each task records its body variant, tensor regions, full reduction requirements, source IDs, participants and effects.
- [ ] Compute read/write regions from composed index maps. Resolve views and aliases to underlying storage. Derive every required data and effect dependency.
- [ ] Use exact pairwise overlap checks for small fixtures. Record construction time and edge count; optimize the algorithm only if those become limiting.
- [ ] Start with explicit producer/consumer sets for events. Count distinct producer tasks, and require one completion contribution per producer/event pair.
- [ ] Verify complete workload coverage, valid bindings, output ownership, reduction completeness and acyclicity. Intentional pure recomputation must be identified; effects cannot be duplicated.
- [ ] Build a CPU simulator of the supported task/event schedule for small graphs. Vary the order and delay of ready tasks to compare its permitted execution with the dependency specification.

The simulator checks compiler bookkeeping. GPU tests remain necessary for memory visibility and hardware progress.

### M4.2 — Simplify readiness events

- [ ] Implement same-consumer-set fusion: merge the events and union their producers.
- [ ] Implement same-producer-set fusion: merge the events and union their consumers.
- [ ] Deduplicate producer/consumer sets, recalculate trigger counts and regenerate links after each change. A producer shared by merged events contributes once to the merged event.
- [ ] Compare required task precedence before and after fusion on small fixtures. Preserve dependency reachability and ensure no unintended waits/cycles appear.
- [ ] Keep flat event/task arrays with offset/count lists. Preserve arbitrary prerequisite/completion lists initially; metadata compression is a later measured optimization.

Use the architecture's two-tile MLP as a hand-checkable example: each multiply tile waits for its two projection tiles; each down-projection tile waits for the mix tiles covering its reduction. After fusion the same conditions must hold. MPK provides the event-fusion reference. [MPK compiler](https://arxiv.org/html/2512.22219v2#S4)

**Checks:** identical producer sets, identical consumer sets, overlapping-but-different sets that must remain distinct, repeated edges, fan-out and a deliberately missing reduction dependency. Compare counts as well as graph edges.

### M4.3 — Execute with static persistent workers

- [ ] Assign a common topological rank to tasks. Build per-worker lists that respect it; include those ordering edges in the progress check.
- [ ] Carry forward M3's verified launch/residency and in-kernel initialization. Start with fixed worker CTAs and no separate scheduler allocation.
- [ ] Implement the loop: observe prerequisites → run body → finish async work → publish results → notify completion → advance. Keep required warp/CTA collectives uniform.
- [ ] Define completion for workers with no tasks or early-finished lists. They must not abandon a later grid collective. Whole-workload completion includes every requested output and effect.
- [ ] Import and execute the complete MLP: gate/up → activation/multiply → down-projection. No downstream work may read an incomplete reduction input.
- [ ] Recompute residency from the compiled entire kernel whenever body variants, threads, pipeline depth or shared-memory use change.
- [ ] Instrument optional per-worker useful-work and wait intervals. Keep profiling off when collecting final latency samples.

**Progress check:** data dependencies plus worker-order edges form an acyclic order; workers required to make progress can run; finite bodies do not hold resources while waiting for another task to release them. The first runtime deliberately excludes cross-task prefetch/resource acquisition so this argument stays small.

### M4.4 — Plan workspace and lifetimes

- [ ] Allocate distinct global slices for cross-worker intermediates first. Keep local scratch contracts separate from globally visible outputs.
- [ ] Track each intermediate's producer and all consumers, including asynchronous uses. Make last-consumer completion the release condition.
- [ ] Add reuse only where a happens-before proof puts every old read before the new write. If adding release dependencies, recheck acyclicity/progress and include their cost.
- [ ] Preserve returned outputs, aliases, per-invocation isolation and same-stream reuse. No hidden clear/reset/copy helper kernels are permitted.
- [ ] Measure peak workspace, shared-memory allocation and transfer/conversion costs alongside latency.

**Checks:** late final consumer, two readers on different workers, attempted early reuse, repeated workspace use, independent invocations and output buffers that remain live after return.

### M4.5 — Select a measured plan and cache it safely

- [ ] Enumerate a bounded set of body variants, tiles, worker counts, pipeline depths and compatible layouts/placements. Set an explicit trial/time budget.
- [ ] Prune invalid semantic, layout, lifetime and progress configurations. Compile survivors and check complete-kernel registers, shared memory, spills and residency.
- [ ] Validate each survivor, then benchmark the complete MLP. Keep the fastest valid incumbent and retain rejection reasons.
- [ ] Cache by graph/body identity, guarded shapes/strides/dtypes, numerical policy, semantic constants, hardware/MIG resources and toolchain versions.
- [ ] Bind new input/parameter addresses at execution. Include an invalidation contract for any folded or prepacked values. A cached plan cannot retain unguarded example-input contents.

**Exit:** a captured complete MLP executes correctly in one persistent kernel using verified tasks/events, a proved static schedule and correct workspace lifetimes. Repeated calls pass, compiled resources support the launch, and tuning results are reproducible.

**Save:** source-to-task coverage, pre/post-fusion graphs, event counts, simulator checks, runtime progress protocol, memory lifetimes, GPU stress tests, worker timelines, candidate table and one-launch trace.

## M5 — Semantic definitions and a complete transformer block

**Purpose:** express common LLM computations clearly and lower them to reusable bodies without losing their exact semantics.

**Depends on:** M4.

**Likely files:** extend generic import/verifiers, add a small semantic-definition/matcher module, and introduce norm/attention bodies as needed. Reuse existing pointwise/view/GEMM code for other work.

### M5.1 — Define semantics from the real block

- [ ] Capture a fresh complete block through M0 on the target device. Inventory exact overloads, tuple outputs, casts, masks, layouts and all live results.
- [ ] Add required generic reductions and missing indexing forms. Decide explicitly how to import or expand pinned post-grad primitives such as the saved trace's `prims.prepare_softmax_online` and `getitem` forms.
- [ ] Create a definition table for RMSNorm, RoPE, Attention and GatedMLP. Each entry has typed parameters, a reference body/expansion, a semantic verifier, a matcher and candidate body families.
- [ ] Match normalized Compute IR. Keep region references and source IDs; do not replace a match with an opaque instruction or hard-code SmolLM layer numbers.
- [ ] Reject only unsupported execution variants. An unmatched semantic pattern may still compile through supported generic operations; an unresolved opaque operation still blocks the whole workload.

For each definition, document one valid example and a nearby variant that must either get different attributes or fail matching. This makes the abstraction testable instead of relying on a name.

### M5.2 — Add bodies with exact numerical contracts

- [ ] **RMSNorm:** prove axes, epsilon location/value, accumulation dtype, weight order and casts. Preserve the captured variant's BF16 rounding before weight multiplication where present.
- [ ] **RoPE:** prove rotation convention, rotated dimension, positions, cos/sin operands, broadcasting, layout and arithmetic order.
- [ ] **Attention:** prove head/group mapping, scale, mask/causality, softmax axis/dtype, probability rounding, dropout and state effects. Test padding and fully masked rows according to the source's behavior.
- [ ] **GatedMLP:** reuse the complete M4 path and verify all activation/cast attributes and any externally used branch outputs.
- [ ] Give new bodies the same capability/completion interface as GEMM. Reuse optimized device implementations only when their target, participation, memory and numerical requirements fit the enclosing kernel.
- [ ] Compare expanded generic semantics and each supported body with the reference. Record tolerances before tuning. Do not fix a failed candidate by silently widening them.

An online-softmax body or an algebraically rearranged norm/projection requires a numerical-policy decision if it changes the captured graph's operations or explicit cast boundaries. Begin with supported faithful implementations; evaluate any relaxed candidates separately. [Mirage RMSNorm/linear inspiration](https://mirage-project.readthedocs.io/en/latest/tutorials/rms-norm-linear.html)

### M5.3 — Compose and tune the whole block

- [ ] Cover both normalizations, Q/K/V projections, RoPE, attention, output projection, full MLP including down-projection, and both residual paths.
- [ ] Generate task/event plans from the chosen bodies. A semantic operation may span tasks; a task may combine compatible portions of several semantic operations.
- [ ] Try concrete local-composition candidates such as norm/projection, compatible Q/K/V work or gate/up/activation. Keep every source cast, side output and effect; reject candidates whose local resources do not fit.
- [ ] Include layout conversions, intermediate storage and scheduler overhead when tuning the assembled block. Do not select a body only because its isolated microbenchmark improves.
- [ ] Use intermediate-value comparisons in diagnostic runs to locate errors, then validate complete unmodified block outputs.

**Exit:** one complete real block passes source-coverage, numerical/effect and one-launch checks for declared shapes and masks. Unsupported variants produce useful source diagnostics. Complete-kernel resources and baseline comparisons are recorded.

**Save:** semantic definitions and valid/invalid matches, generic expansions, chosen body capabilities, coverage map, task/event plan, numerical results and full-block timing.

## M6 — Full SmolLM inference in one megakernel

**Purpose:** meet the full-model execution target, including work surrounding the transformer blocks.

**Depends on:** M5. Start with the existing explicit no-KV-cache inference regime and then a larger prefill shape.

### Build it

- [ ] **Capture the complete invocation.** Pin the model revision and requested outputs. Include embedding, captured position/mask work, every block, final norm, output projection and all observable effects. Tokenization may be outside the declared tensor workload; captured tensor work cannot be moved outside just to hide launches.
- [ ] **Close the coverage report.** Add required generic semantics/bodies outside the block. Every source operation must be implemented or eliminated by a verified semantics-preserving transformation. An unresolved `OpaqueFX` region fails compilation.
- [ ] **Reuse bodies across layers.** Bind repeated body implementations through task descriptors and parameters. Measure dispatch overhead and generated code size; avoid replicating the entire body for every layer without a reason.
- [ ] **Preserve runtime bindings.** Check current weights, tied parameters, output trees, guards, aliases and state. No baked-in example addresses or values without an explicit valid constant contract.
- [ ] **Plan model-scale memory.** Derive lifetimes across workers/layers, preserve live outputs and recheck compiled register/shared-memory limits and residency for the complete megakernel.
- [ ] **Check complete outputs.** Compare full requested tensors, not only argmax tokens, against eager and an equivalent strong Inductor baseline. Use multiple seeded inputs, applicable masks and both selected shape regimes.
- [ ] **Audit the entire call for launches.** Inspect warmed execution including frontend wrappers. Count reset, initialization, conversion, copy-back and helper kernels if present. Include required work in the one kernel or reject the unsupported behavior. Report genuine compile/setup separately.
- [ ] **Optimize the measured full-model critical path.** Investigate device-body throughput, waits, memory traffic, conversions, spills and code size. Keep a faster validated incumbent while evaluating changes.

The architecture's one-launch requirement includes runtime preparation that must recur for each invocation. The inspected MPK source has preparation and optional split scheduler/worker paths, so adopting its launcher unchanged would not establish our contract. [MPK launch implementation](https://github.com/mirage-project/mirage/blob/mpk/include/mirage/persistent_kernel/persistent_kernel.cuh#L1851)

**Exit:** the complete supported SmolLM invocation produces correct outputs/effects through exactly one workload kernel, with reproducible latency and resource measurements. Report supported shapes and numerical policy explicitly. A slower result proves model correctness while leaving a documented performance gap.

**Save:** model/workload identity, complete coverage, generated plan/source hashes, numerical errors, state/alias checks, resource/workspace/code-size reports, whole-call launch trace and baseline timing samples.

## M7 — Stateful workloads and measured optimization

**Purpose:** extend supported workloads and reduce the bottlenecks observed in M6. This stage has an ordered core and independent optimization experiments; it is not a requirement to implement every advanced technique.

**Depends on:** M6. Implement M7.1 for stateful decode, then choose M7.2–M7.5 from evidence. Each experiment retains a correctness-checked previous configuration for comparison.

### M7.1 — Add explicit prefill/decode and KV state

- [ ] Capture actual prefill and decode graphs and define their invocation boundaries. One decode step initially means one complete step in one kernel; a loop over many generated tokens is a separate declared workload.
- [ ] Represent KV reads/writes, valid lengths, positions, aliases and update order. Include required cache/state updates inside the megakernel.
- [ ] Validate consecutive decode steps against the reference, including cache reset/reuse, length boundaries, stale data and changed requests.
- [ ] Add concrete guarded specializations for needed batch sizes, sequence lengths and strides. If using buckets, prove padding/masking and cache-validity behavior; do not infer semantic correctness from a bucket size alone.

**Gate:** every declared regime passes full output/state correctness, specialization checks and one-launch auditing. A short no-cache forward still does not count as decode.

### M7.2 — Add hybrid dispatch when work durations vary

**Trigger:** worker timelines show static scheduling leaves usable compute capacity idle because task durations vary.

- [ ] Keep static lists for predictable regions. Add ready-task dispatch to regions with measured imbalance; a region boundary returning to static scheduling must actually remove or account for that imbalance.
- [ ] Give ready dynamic work priority over an unready static head. Define unique task claiming, queue publication and how new ready work is discovered.
- [ ] Specify queue bounds, overflow/backpressure, fairness and termination with queued or in-flight work. Include scheduler threads/CTAs in the residency and progress analysis.
- [ ] Compare worker-only and reserved-scheduler arrangements only as needed. Tune on the actual GPU/MIG resources rather than copying a scheduler-to-worker ratio.
- [ ] Stress deliberately unequal task durations, multiple ready producers, queue capacity and termination races. Revalidate visibility and repeated invocation.

**Gate:** correctness/progress hold and full-workload latency improves for the targeted regime. Retain static dispatch where it wins. MPK's hybrid dispatch motivates this experiment. [MPK runtime](https://arxiv.org/html/2512.22219v2#S5)

### M7.3 — Compress task/event metadata when it costs time

**Trigger:** descriptor traffic, event processing or task-graph size is material in profiles.

- [ ] First measure flat offset/count lists and reuse shared body parameters to avoid needless duplication.
- [ ] Consider normalizing tasks to one prerequisite and one completion event using verified relay tasks/events where needed.
- [ ] Only after the graph supports it, reorder tasks so an event's successors occupy a contiguous range. Preserve explicit lists for sets that cannot use this representation.
- [ ] Compare reachability, unique execution and event counts with the uncompressed graph in the simulator and GPU fixtures. Include relay overhead in timing and preserve source provenance.
- [ ] Optionally prefetch descriptors when their addresses are known and their contents immutable for the invocation.

**Gate:** lower complete-workload latency or a necessary reduction in graph memory, with the tradeoff recorded. Fewer metadata bytes alone do not prove faster inference. [MPK task/event records](https://github.com/mirage-project/mirage/blob/mpk/include/mirage/persistent_kernel/runtime_header.h)

### M7.4 — Overlap loading with computation

**Trigger:** profiles show memory-pipeline bubbles between tasks on the same worker.

- [ ] Add an optional preload/compute split to the bodies being tested. Keep the existing run-to-completion interface for other bodies.
- [ ] Begin with one worker and bounded double buffering. Preload only ready input regions; distinguish independent weights from activations whose producers have not completed.
- [ ] Track live shared-memory slices, async/barrier state and alignment across both tasks. Make prefetch optional if space is unavailable; it must never block the current task's completion.
- [ ] Release storage only after all corresponding async accesses finish. Verify behavior with delayed producers, tails and repeated body transitions.
- [ ] Add paged shared-memory allocation only if variable body footprints justify it. Define allocation order, release rules and a progress argument before enabling overlap.
- [ ] Compile the full kernel again and measure register pressure, shared memory, spills, occupancy and full latency with prefetch on/off.

**Gate:** the intended workload becomes faster while preserving dependencies, numerical behavior and progress. More overlap is not an objective by itself.

### M7.5 — Expand search and model coverage deliberately

- [ ] Use measured bottlenecks to add body/tile/layout/recompute candidates, maintaining a finite search budget and cached validated incumbent.
- [ ] If evaluating algebraic rewrites or approximate arithmetic, use an explicit numerical-policy specialization, validate operation and full-model errors, and keep required effects and cast boundaries accounted for.
- [ ] Add models by inventorying missing generic semantics and state first. Implement needed scan/convolution/scatter or other bodies before claiming support; model-specific dimensions remain parameters.
- [ ] Extend the results matrix across supported workloads, GPUs/toolchains and numerical policies. Record compile/tuning cost, memory, errors and effective baselines.
- [ ] Re-run affected regressions after shared body, scheduler, lifetime or cache changes. Retain the best measured valid configuration per declared regime.

**Gate for each addition:** full correctness/state behavior, valid specialization, one workload launch and reproducible performance evidence. Global optimality is not claimed from finite search.

Multi-GPU communication, general superoptimization, solver-based global layout search and a full serving/request scheduler are future scope decisions. They can become appropriate with concrete workloads; they are not prerequisites for the initial optimized single-GPU model megakernel.

## Keep a completion record for every stage

Store a small record with the corresponding artifacts:

```text
Stage and exact declared workload:
Source revision / graph, body and plan hashes:
Model revision / input shapes, strides, dtypes and output scope:
Numerical policy and reference tolerances:
GPU / MIG resources / toolchain versions:
Implemented changes and actual commands:
Correctness, state, alias and specialization results:
Task/event and progress checks:
Steady-state workload launch count:
Compiled resources / workspace / generated code size:
Raw timing samples and effective baselines:
Rejected candidates and reasons:
Exit gate passed or failed, with evidence:
Remaining performance gaps and next task:
```

Tests and benchmark commands are created during their stages. Once present, run the focused checks using the repository environment, for example `.venv/bin/python -m pytest tests/test_frontend.py` if that is the chosen file. Save the actual commands that passed; the example does not imply that the test already exists.

**First implementation task: M0's version-pinned live handoff and cache/runtime checks.** The hardest early kernel task is M3's in-kernel initialization and cross-CTA visibility/progress proof. The first full-model success gate is M6. The optimization loop continues against the complete workload throughout and after those milestones.
