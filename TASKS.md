# MegaBake build tasks

**Status: open implementation tasks. Existing research results do not complete these tasks.**

Use this checklist to choose the next piece of work. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) defines the acceptance gates. [ARCHITECTURE.md](ARCHITECTURE.md) defines the contracts. [north-star.md](north-star.md) fixes the scope.

## How to use this list

- Follow the numbered sections in order. Complete one runnable case before adding more cases.
- Mark a task complete when its code and required checks work. Keep the command and result with the task evidence.
- Record correctness, launch count, and performance separately for each workload and GPU target.
- If a gate fails, fix the measured cause before expanding coverage. Keep the failed result.
- Use the optimization tasks whenever a gate needs them. They can become necessary during the first MLP experiment.
- Build production code under `src/megabake`. Add files as working tasks need them.
- Keep reusable checks under `tests` and exploratory probes under `experiments`.

For each component, start from the matching Mirage implementation in [SCHEDULER_REUSE.md](SCHEDULER_REUSE.md). Follow its callers, assumptions, and tests. Preserve compatible behavior. Record the source revision, adaptation, comparison result, and required attribution with each port.

### Build order

| Tasks | Gate | Result required before expansion |
| --- | --- | --- |
| 01–03 | G0 | Fast device bodies execute together with valid synchronization and resource use |
| 04 | G1 | A complete MLP beats the strongest equivalent measured baseline |
| 05–09 | G2 | FX produces working megakernels, including generic coverage and explicit fallback |
| 10–12 | G3 | A complete transformer block works in prefill and stateful decode |
| 13–14 | G4 | The complete declared model invocation passes validation and performance checks |
| As needed | G5 | A targeted change removes a measured bottleneck |

### Existing starting points

| File | Use |
| --- | --- |
| [verify_smollm_fx.py](src/eager/verify_smollm_fx.py) | Current capture and reference logic; inspect its cache and post-grad handoff |
| [dump_smollm_fx.py](src/transformers/dump_smollm_fx.py) | Small wrapper around that entry point |
| [probe.py](experiments/design_2026_10_08/probe.py) | GEMM measurements and source pins |
| [restart_body.py](experiments/design_2026_10_08/restart_body.py) | Experimental body adapter; compare its restart behavior with a retained pipeline |
| [mlp_baseline.py](experiments/design_2026_10_08/mlp_baseline.py) | Complete MLP reference and baseline measurements |
| [check_memory_planner.py](experiments/reuse_2026_10_09/check_memory_planner.py) | Isolated check of Mirage's allocation core |

These files provide starting evidence. Recheck them on the selected target before using their results for a gate.

## 01. Fix the first workload and measurement setup

**Gate: G0. Start here.**

- [ ] Select the GPU target and pin the software versions. Record the actual SM count and cooperative-launch support.
- [ ] Pin the first model checkpoint. Declare outputs, precision, numerical tolerances, and required state updates.
- [ ] Extract actual projection shapes and strides. Include gate/up/down, Q/K/V, output projection, and the vocabulary head.
- [ ] Select small-M and larger-M fixtures, partial tiles, and weight sets below and above L2 capacity.
- [ ] Record separate prefill and stateful decode fixtures. Keep their shapes and cache requirements available from the start.
- [ ] Reproduce library GEMM and complete MLP baselines. Record effective Inductor choices and whether autotuning ran.
- [ ] Make one measurement harness report numerical errors, raw timings, launch count, and compiled resource use.
- [ ] Separate setup, compilation, first-call, warm GPU, and complete host-call timing. Include a weight-streaming case.

**Deliverable:** a workload record and repeatable baseline commands with saved results.

## 02. Make a fast GEMM body process a range of tiles

**Gate: G0. Requires task 01.**

- [ ] Select a measured official CuTe GEMM configuration for each initial regime. Retain its original launch as a reference.
- [ ] Extract device work behind the proposed `run(work_iterator, tensor_bindings, scratch, epilogue)` interface.
- [ ] Keep the pipeline active across the assigned tile range. Allocate reusable scratch in the enclosing worker.
- [ ] State thread participation, warp roles, barrier state, output readiness, and scratch-release conditions.
- [ ] Bind current addresses and descriptors on each call. Check repeated entry with changed tensors, contents, and tile counts.
- [ ] Compare the original launch, the range entry, and the restarting adapter. Record latency and resources for each.

**Deliverable:** a correct GEMM range body with measured composition cost and an explicit entry/exit contract.

## 03. Run different bodies inside one persistent kernel

**Gate: G0. Requires task 02.**

- [ ] Generate a small vector body and a reduction body from typed expressions. Include different casts or scalar parameters.
- [ ] Run GEMM, generated work, and the next required GEMM configuration inside one kernel.
- [ ] Start with a manually specified phase plan and distinct intermediate buffers.
- [ ] Implement and test the actual cooperative launch and grid barrier in CuTe DSL. Include workers with no assigned tiles.
- [ ] Derive the worker limit from the compiled complete kernel. Reject an oversized launch before execution.
- [ ] Check cross-CTA data visibility, asynchronous completion, scratch reuse, and transitions between warp roles.
- [ ] Keep per-call state private. Include required initialization inside the workload kernel.
- [ ] Test repeated calls, changed bindings, delayed producers, and the largest valid worker grid.
- [ ] Run the relevant memory/race checks. Save the launch trace and complete resource report.

**Deliverable:** a reproducible mixed-body executable that passes G0. Resolve body or synchronization failures before task 04.

## 04. Complete and measure the first MLP

**Gate: G1. Requires G0.**

- [ ] Compose gate/up projections, SiLU, multiplication, and down projection. Preserve every source cast and requested output.
- [ ] Add the full producer barrier before the down projection. Verify that all reduction inputs are ready.
- [ ] Compare separate projections, a valid packed projection, and paired local work where supported.
- [ ] Compare local epilogue fusion with explicit intermediate work. Count packing, conversion, reset, and workspace costs.
- [ ] Measure eager, CUDA Graph replay, effective `torch.compile`, and the complete megakernel under equivalent conditions.
- [ ] Repeat the small-M and larger-M cases separately. Use alternating measurements when results are close.
- [ ] Identify the dominant loss for each failed case. Use the optimization tasks if phase waits or work assignment dominate.
- [ ] Save a pass/hold report for each regime. Require an improvement beyond measurement noise before expanding that regime.

**Deliverable:** a complete MLP performance report with a measured one-kernel result. Passing one regime does not pass the other.

## 05. Establish the live FX handoff

**Gate: G2. Requires G1 for the selected regime.**

- [ ] Extract a small version adapter from the current capture path. Check the supported PyTorch build before importing private APIs.
- [ ] Receive live FX after the selected post-grad passes and before Inductor lowering or scheduling.
- [ ] Repair the cache path. Check that compilation receives the intended graph phase on both relevant cache paths.
- [ ] Preserve argument bindings, guards, aliases, mutations, output structure, and the expected callable wrapper.
- [ ] Start with a reference callable to check the handoff. Keep its execution type explicit.
- [ ] Test changed inputs, parameter mutation, tensor replacement, and guard failures. Reject unsupported training or graph breaks.

**Deliverable:** a tested frontend boundary that can hand a live graph and runtime contract to MegaBake.

## 06. Import computation and generate baseline bodies

**Gate: G2. Requires task 05.**

Implement tasks 06–08 around the same small fixture. Add one operation family, compile it, and check the result before expanding.

- [ ] Define the Compute Graph values, operations, regions, and source references described in the architecture.
- [ ] Import shapes, strides, types, aliases, effects, and explicit numerical rules. Retain unknown operations with their FX source.
- [ ] Represent indexed loads, reductions, typed scalar expressions, predicates, and stores as tensor programs within operations.
- [ ] Import maps/casts, views/copies, contractions, and reductions. Add tuple selection, indexing, and state access when fixtures need them.
- [ ] Add selected PyTorch decompositions. Preserve source casts, effects, and useful contraction or attention structure.
- [ ] Generate baseline map, reduction, and indexed-contraction bodies using the interface proved in G0.
- [ ] Handle supported broadcasts, strides, partial tiles, and degenerate dimensions. Use bounded loops or partial-reduction phases when needed.
- [ ] Check generated results against the original graph with named region matchers disabled.

**Deliverable:** generic device generation for the fixture's primitives, with clear errors for missing semantics.

## 07. Turn the graph into an executable kernel plan

**Gate: G2. Build alongside task 06.**

- [ ] Define KernelPlan with body configurations, tile maps, phase order, dependencies, storage, bindings, and launch configuration.
- [ ] Adapt Mirage's relevant tile/layout and copy checks. Compare generated offsets with a simple coordinate reference.
- [ ] Derive producer/consumer dependencies from actual accesses. Include aliases, state ordering, and every reduction input.
- [ ] Emit an ordered phase plan with the runtime proved in G0. Preserve all graph dependencies, including forks and joins.
- [ ] Allocate distinct global intermediate slices first. Reuse scratch only after all users and asynchronous operations finish.
- [ ] Adapt Mirage's allocation core when reuse is needed. Supply verified lifetimes and padded sizes.
- [ ] Emit one complete CuTe kernel and check its compiled resources before launch.
- [ ] Return the generated callable through the frontend wrapper. Bind live inputs and parameters on every call.
- [ ] Dump source operations, body choices, phase order, storage, launch settings, and rejection reasons.

**Deliverable:** FX-to-megakernel execution for the proven fixture, with results comparable to its manually specified G1 plan.

## 08. Add tuned selection and cache measured plans

**Gate: G2. Requires a working path through tasks 06–07.**

- [ ] Register baseline and tuned generators behind one provider interface. Return valid configurations or specific rejection reasons.
- [ ] Match the MLP region while retaining its source operations and every live output.
- [ ] Adapt relevant Mirage candidate-generation rules. Keep tile, instruction, layout, stage, and warp choices coupled within body families.
- [ ] Search a small set of complete plans. Check correctness and compiled resources before retaining timing results.
- [ ] Cache the selected plan using the complete key in architecture section 11. Include strides, types, numerical policy, and state layout.
- [ ] Keep tensor addresses out of reusable plan identity. Rebind live parameters and invalidate stale packed or folded data.
- [ ] Test cache reuse, guard changes, parameter replacement, and mutation. Recheck latency and launch count against G1.

**Deliverable:** automatic body selection and repeatable plan reuse without stale runtime bindings.

## 09. Prove generic coverage and explicit fallback

**Gate: G2. Requires tasks 06–08.**

- [ ] Compile norm variants with changed epsilon, affine/residual terms, and intermediate casts. Disable named matchers for this check.
- [ ] Compile a small vision fixture with convolution, activation, pooling, and a residual. Add the required indexed import rules.
- [ ] Check padding, stride, dilation, and groups. Generate from operation meaning without registering the model.
- [ ] Compile unfamiliar primitive combinations and a fork/join graph that MPK's single-event format cannot represent.
- [ ] Add an opaque custom operation with no lowering. Report its FX source and exact missing rule before launching work.
- [ ] Implement `fallback=error|inductor`, with `error` as the default. Choose the whole-invocation path before device execution.
- [ ] Test delegation without recursive MegaBake entry. Preserve outputs, aliases, mutations, guards, and changed runtime inputs.
- [ ] Check that a device failure cannot trigger fallback after a stateful invocation has partly executed.
- [ ] Report coverage, generic-body use, execution kind, correctness, launch count, and performance separately.

**Deliverable:** G2 evidence for unfamiliar supported graphs and explicit unsupported cases. External fallback does not count as a megakernel pass.

## 10. Generate norm, residual, and RoPE variants

**Gate: G3. Requires G2.**

- [ ] Import exact definitions from real captures. Keep axes, epsilon placement, casts, rotation rules, and extra outputs explicit.
- [ ] Generate scalar expressions around reusable vector and reduction schedules.
- [ ] Add optional tuned matchers. Keep the generic path available when a variant fails a matcher.
- [ ] Check uneven widths, supported strides, and multiple variants against the source graph.
- [ ] Compose these bodies with the MLP. Repeat G0 resource, transition, and synchronization checks for the expanded mixture.

**Deliverable:** generated norm and RoPE variants that work inside the existing megakernel.

## 11. Add composable attention

**Gate: G3. Requires task 10.**

- [ ] Measure the selected attention implementation on the target. Inspect relevant Mirage Hopper bodies and their tests.
- [ ] Extract or port its device work while retaining the tuned tile pipeline and scratch contract.
- [ ] Generate supported typed score and mask expressions inside that pipeline. Check access and numerical restrictions for each hook.
- [ ] Preserve grouped heads, causal offsets, probability casts, softmax policy, and every requested output.
- [ ] Test partial tiles, fully masked rows, and poisoned scratch. Compare repeated body entry with the original implementation.
- [ ] Route unsupported hooks through generic primitives when valid. Account for score-tensor storage and workspace limits.
- [ ] Compose attention with the projection and norm bodies. Measure the complete mixture's resources and transition costs.

**Deliverable:** a checked attention family and an explicit path for variants outside its tuned interface.

## 12. Complete prefill and stateful decode blocks

**Gate: G3. Requires task 11.**

- [ ] Add `--mode prefill|decode` and equivalent backend configuration. Validate each mode against the captured workload.
- [ ] Select scheduling policy from actual shapes, state, and target resources. Keep computation defined by the graph.
- [ ] Add a bounded contiguous KV cache with explicit read/write domains and ownership.
- [ ] Include cache updates and all requested outputs in the workload kernel. Guard capacities and runtime valid lengths.
- [ ] Compose the full transformer block, including attention, projections, norms, residuals, and MLP.
- [ ] Check complete prefill and at least two advancing decode steps. Include repeated calls near the cache capacity boundary.
- [ ] Compare both outputs and cache contents with the native model. Include causal offsets and supported mask variants.
- [ ] Measure each block regime against an equivalent baseline. Fix losses that exceed the available savings before expanding.

**Deliverable:** a complete block report for prefill and stateful decode, with separate correctness and performance results.

## 13. Compile the complete model invocation

**Gate: G4. Requires G3 for the relevant regime.**

- [ ] Use the pinned SmolLM fixture. Include embedding, every layer, final norm, vocabulary projection, and the requested outputs.
- [ ] Add missing gather/index or state rules through the generic importer. Preserve exact index behavior and dependency coverage.
- [ ] Reuse equivalent body code across layers. Bind layer parameters through data instead of duplicating bodies for tensor addresses.
- [ ] Keep all masks, aliases, mutations, and KV updates required by the captured invocation.
- [ ] Check complete-kernel resources, code size, workspace, and worker residency after adding all body families.
- [ ] Test changed inputs, changed weights, and advancing decode. Use intermediate comparisons to locate errors.
- [ ] Trace the complete callable. Include wrapper work and all recurring initialization, conversion, and copy-back work.

**Deliverable:** the complete declared model invocation in one workload kernel, with correct outputs and state.

## 14. Establish completion for the declared targets

**Gate: G4. Requires task 13.**

- [ ] Run the complete target matrix from task 01. Report prefill and decode independently.
- [ ] Compare with equivalent `torch.compile` and other measured stronger baselines. Match outputs, precision, state, GPU partition, and warmup.
- [ ] Record complete-call latency, GPU latency, raw samples, numerical errors, launch traces, resources, and peak workspace.
- [ ] Report compilation, setup, packing, and first-call costs separately. Include any recurring work in steady-state timing.
- [ ] Require a complete-model improvement beyond measurement noise for each target claimed as a performance success.
- [ ] Keep correct but slower targets visible as supported cases with a performance gap. Keep unsupported cases explicit.
- [ ] Add a documented command to compile, check, and benchmark each declared target from a fresh process.
- [ ] Run retained regression fixtures for variants, generic coverage, fallback, cache invalidation, aliases, and repeated stateful calls.
- [ ] Publish the support table and update implementation status. Link each completion claim to its evidence.

**Deliverable:** a reproducible compiler path and a complete-model result for each declared target. Open performance gaps remain open work.

## Optimization tasks: use when a gate needs them

These tasks belong to G5. Apply one at the gate where its cost is measured. Return to that gate after checking the change.

- [ ] **Phase waits:** derive exact tile dependencies and port suitable MPK event grouping. Check against concrete read/write overlap.
- [ ] **Event runtime:** preserve arbitrary prerequisite sets. Prove visibility, unique contributions, and progress with worker order included.
- [ ] **Event encoding:** test forks, joins, residuals, reductions, and uneven tails. Retain the phase plan when compression cannot encode them.
- [ ] **Work imbalance:** start with improved static assignment. Add bounded dynamic claiming only when measurements justify its extra state.
- [ ] **Too few GEMM tiles:** compare smaller tiles, split K, or Stream-K. Include partial storage and numerical checks.
- [ ] **Weight stalls:** add prefetch between compatible ranges. Check input readiness, buffer ownership, and asynchronous completion.
- [ ] **Workspace pressure:** reuse storage only after proved release. Recheck all asynchronous readers and concurrent-call isolation.
- [ ] **Decode attention:** split the context when useful. Include stable partial statistics, merge cost, and state ordering.
- [ ] **Cluster opportunity:** add a cluster body only with valid collective scope, occupancy, and compatible launch settings.
- [ ] **Code growth:** reduce duplicate variants or repeat layer phases through loops. Measure compile time and instruction-cache effects.

For each change, keep a before/after report for the complete affected workload. Retain it only when it passes correctness and improves the measured target.
