# MegaBake V3: FX, indexed semantics and plan contracts

Status: normative proposed IR design, revised 2026-09-27. The [architecture](MEGABAKE_V3_ARCHITECTURE.md) owns scope; this file owns representations, semantics and verification obligations. Schemas are contracts, not claims of implemented Python classes.

## 1. Representations and their purpose

| Representation | What it adds | What it must preserve |
|---|---|---|
| `CapturePackage` | FX/ExportedProgram, bindings, example inputs, guards, numerical and state contract | Complete requested invocation and source provenance |
| Normalized FX + `TensorFacts` | Canonical selected ATen forms, shapes, strides, aliases, effects and cast facts | Executable original reference and all live values |
| `IndexedTensorProgram` | Typed maps, reductions, contractions, indexing, scans/control, state effects and verified repeated regions | Exact FX meaning or guarded declared numerical equivalence |
| `AlgorithmChoices` | Bounded equivalent algorithms, composites, layouts and preparation options | Reference expansion and guard for every choice |
| `LogicalExecutionPlan` | Parametric tile domains, access maps, reductions, dependencies, movement opportunities and lifetimes | Indexed program behavior independent of device execution names |
| `TargetExecutionPlan` | Body tactics, physical layout/storage, worker schedule, synchronization, invocation and artifact | One target's correct realization of the logical obligations |

`AlgorithmChoices` may be an overlay on the indexed program rather than a separate graph library. `TensorFacts`, body measurements and `TargetProfile` are analyses. The logical and target plans may share implementation records, but a logical plan cannot masquerade as an admitted target artifact. FX remains the executable oracle for transformed regions.

**FatOps cease to be the canonical IR.** A named pattern is one rule of the form `match + guards + exact reference + equivalent algorithm candidates`. Ordinary supported indexed operations have a generated device-body route without a pattern. A new family that changes the arrangement of those operations therefore does not require a new model-specific FatOp or kernel. An opaque custom operation, novel recurrence or unsupported data-dependent effect requires an explicit semantic contract; an optimized implementation may require a new reusable algorithm provider.

## 2. Capture and the Inductor handoff

A complete `ExportedProgram` is preferable when it preserves graph signature, lifted parameters, state and constraints. A bare `GraphModule` is accepted with explicit example inputs, parameter/state bindings, output structure and mutation assumptions. `torch.compile` custom backends may receive only graph-break fragments; a fragment is not a full-step megakernel even if its own compilation succeeds. The public direct-FX path is the primary full-step entry.

Take the canonical handoff at functional FX/ATen before Inductor `GraphLowering` turns the program into loop/buffer/external-kernel objects. There is no stable universal boundary described as “all Inductor optimization but before fusion.” The [PyTorch 2.6 compile flow](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/compile_fx.py) runs several graph passes before `GraphLowering`; its [post-grad pass](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/_inductor/fx_passes/post_grad.py) can reorder, eliminate or reinplace operations. These transforms have preconditions that a bare FX graph with cache effects may not satisfy. Reuse is explicit rather than an all-or-nothing Inductor handoff:

| PyTorch/Inductor component | V3 use | Admission boundary |
|---|---|---|
| `ExportedProgram` graph signature, range constraints and `run_decompositions` | Primary lifted-binding/effect ABI and selected inference ATen normalization | Retain original executable reference; compare graph signature, state effects, output tree, casts and origin mapping after each selected decomposition. Keep exact attention/recurrence forms when decomposition would erase useful algorithm choices. |
| Fake tensor metadata, `SymInt`/shape constraints and operator schema | Seed shape/stride/dtype and symbolic guards | Treat metadata as facts only where validated; prove alias, mutation and pointer alignment separately. No CUDA query during common analysis. |
| Selected Inductor FX rewrites | Optional pure-region CSE, folding or layout/normalization candidates | Pin PyTorch version and individual pass/config; run on a copy; verify live origins, effects and numerical boundaries against the original. Do not invoke an entire `pre_grad_passes`/`post_grad_passes` pipeline as an undocumented normalization switch. |
| Inductor `Pointwise`/`Reduction` loop IR | Experimental source for indexed maps/reductions on isolated pure regions | Translate only expressions whose exact index maps, dtypes/casts, guards and FX origins can be recovered; compare with the local FX evaluator. Reject opaque buffers/extern calls rather than declaring strict coverage. Direct ATen lowering remains available. |
| Inductor templates, generated Triton/CUDA kernels, fusion groups and autotune results | Exact-shape algorithm/layout teacher and matched `ExternalPlan` or baseline | Record selected tactic and measured resources/latency, then generate or adapt a separate device-callable body and remeasure it inside the owner entry. A launchable kernel or autotune cache winner is not an in-grid routine. |

The bounded pure-loop bridge is an experiment, not a reason to import Inductor's graph-wide `GraphLowering`, scheduler, runtime or virtualized state into the portable semantic IR. Inductor's [2.6 `Pointwise`/`Reduction` records](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/ir.py) show reusable indexing structure, while `GraphLowering` also chooses materialization and external kernels. The bridge earns use only if it reduces primitive-lowering work without losing V3's state/provenance contract. New PyTorch versions require a renewed adapter audit. [PyTorch 2.6 export/decomposition API](https://docs.pytorch.org/docs/2.6/export.html)

The `CapturePackage` records:

```text
source FX or ExportedProgram and operator/library versions
original executable reference; graph signature; lifted weights/constants
user inputs, state inputs/updates, outputs and output ownership
example inputs, symbolic shape constraints and specialization guards
numerical policy: dtype, accumulation, casts, reassociation/tolerances
workload: batch/context/cache layout, target device, setup contract
```

A Hugging Face helper may create a fixed-capacity cache, flatten outputs and capture a full step. It returns the graph and binding contract, not a model-name certificate. Config values and module paths are hints checked against the actual graph. Strict compilation reports each unsupported node, effect or guard; ordinary fallback remains visible.

## 3. Ordered semantic analysis, then iterative search

The ordered steps establish trusted facts. Later performance decisions feed back into one another.

1. Validate capture completeness, state/output signatures, supported devices and inference assumptions.
2. Functionalize only effects whose original meaning can be retained; make cache updates and alias constraints explicit.
3. Apply selected decompositions while preserving high-level attention, recurrence or grouped computation as candidate regions when their exact meaning is known.
4. Propagate shape/stride/alias facts and numerical boundaries; perform effect-aware simplification and DCE.
5. Construct the indexed program from supported ATen primitives; retain origin nodes and executable reference expansion for each region.
6. Verify repeated/conditional regions from the graph and bindings; attach iteration-specific weights, state and variant tags.
7. Enumerate guarded algorithm/layout/composite choices without choosing one cover.
8. Generate target body/tile candidates, then derive exact tile dependence and whole-region physical schedules. A failed body/resource/latency result may revise choices from step 7 onward.

A semantic rewrite is admitted by proof or differential evidence under its numerical policy, never because it decreases FX node count. Unsupported regions retain their original reference and a diagnostic. Source graph normalization and backend schedule search are not one irreversible pipeline.

## 4. Indexed tensor and effect semantics

A minimal operation record is:

```text
IndexedOp {
  origin_fx_region, reference_expansion,
  iteration_domain, input_index_maps, output_index_map,
  expression_dag, reductions_or_scan, masks_and_bounds,
  dtype_and_cast_points, layout_constraints,
  alias_reads_writes, observable_effects, numerical_policy
}
```

The vocabulary covers pure elementwise maps, broadcasting, reductions, contractions, views/reshapes with exact indexing, gather/scatter and functional state updates. Bounded branches and recurrences have explicit predicates, state transition and reference meaning. Generic lowering of a feature may be correct yet slow; each operation has a declared strict-support status. This small structured layer borrows indexing-map and iteration-domain ideas from [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) without requiring a general MLIR migration.

For a usual projection with weights stored `[N,K]`:

```text
Y[b,n] = cast_out( sum_{k in K} cast_acc(X[b,k] * W[n,k]) + bias[n] )
```

The record also states whether bias is present, when accumulation rounds, and which transposes/views are logical. The algebraically equivalent matrix formulation `Y^T = W X^T` is an `AlgorithmChoice`, not a change to the reference. If different evaluation order violates the numerical policy, the candidate is rejected or given an explicit tolerance. A stable weight pack is a separately costed preparation action; a dynamic activation pad/transpose is timed per invocation.

A norm retains axes, epsilon placement, weight convention and cast order. A gating expression retains which operand is activated and all intermediate casts. Attention retains scale, masks, causal/window alignment, head mapping, cache valid length and any state effect. `SDPA` recognition can propose online softmax or split context, but its exact FX expansion remains available. A gated recurrent update is a scan/state transition, not an attention label inferred from a config string.

### Effects and aliases

Views can be zero-copy but are not identity; zero strides, storage offsets, transposes and overlapping aliases enter index and lifetime analysis. A cache write remains required even if its returned tensor is not used. A candidate that fuses or reorders operations must preserve every other live output and observable effect. Reusing an allocation in-place requires proving that all old-value readers, including asynchronous operations, have retired.

## 5. Guarded algorithms and repeated regions

`AlgorithmChoice` contains `origin`, semantic guard, exact reference relation, supported numerical policy, prepared layouts, algorithm parameters and an expansion back to indexed work. Choice families include:

| Region | Candidate algorithms |
|---|---|
| Projection | K-parallel GEMV; output-channel-major padded tensor-core work; split-K; tiled GEMM; epilogue fusion |
| Shared-input projections | Separate versus packed QKV or gate/up with exact output partition and casts |
| Attention | Online softmax, split context plus valid combine, conservative indexed expansion |
| MLP | Full hidden materialization versus paired hidden chunks and legal down-projection continuation |
| Norm | Compute once/materialize, local reuse, guarded fused epilogue or measured pure recomputation |
| MoE | Scatter/gather and segmented/grouped contraction with data-dependent domain when supported |

These are alternatives, not required transformations. A larger fused semantic region may still emit multiple logical tasks; several indexed operations may share one device body. The final plan records complete semantic coverage, duplicate pure work when deliberately costed, and every live output/effect.

A `RepeatRegion` is executable structure within the indexed program:

```text
RepeatRegion {
  bound_or_guarded_count, iteration_variable,
  per_iteration_parameter_bindings, per_iteration_state_bindings,
  carried_values_and_effects, variant_predicates,
  body_reference, entry_and_exit_values
}
```

Promote a structural `LayerSummary` only after verifying the graph. Different layers may have different masks, widths, recurrent branches or state contracts. Looping is an implementation option; weight addresses can be known early even when next-layer activations are not. The region makes that dependency and the whole-region liveness visible to CUDA and future TPU planners.

## 6. Parametric logical task plan

A logical task family has a guarded iteration domain, exact read/write region maps, reduction participation, effect constraints and a reference computation. Runtime values may determine a bounded domain cardinality, such as valid context or routed expert count, if the backend supports a safe schedule. Derive producer/consumer relations symbolically from region overlap and effects before materializing events; compress equivalent relations when lowering. This follows the useful level of [Luminal's block expressions](https://blog.luminal.com/p/compiling-models-to-megakernels) and [MPK's event analysis](https://arxiv.org/html/2512.22219v2) without requiring either runtime.

```text
LogicalExecutionPlan {
  indexed_program_hash, guards, numerical_and_state_contract,
  semantic_coverage: [origin_region -> algorithm/task families],
  task_domains: [parameters, bounds, read/write/reduction/effect maps],
  dependencies: [exact producer relation, consumer relation, needed region],
  readiness: [address, input data, compute, publication, finalization],
  movement_opportunities: [source, destination role, earliest legal issue],
  storage_constraints: [extent, alignment, reachability, partial-order lifetime],
  placement_relations: [same owner, local group, collective, independent],
  verification_report
}
```

This plan records legal work and partial order. It does **not** contain CTA/warp/SM counts, `cuda:shared`, TPU VMEM, physical cohort splits, fixed prefetch depths, queue policy or mandated grid joins. A target may propose a different tile granularity and regenerate the affected logical plan before final verification. A tile never publishes more than it has computed; a reduction output is final only after every required contribution and cast.

Distinct conditions include address known, source data ready, movement issued, source access retired, destination visible, compute complete, accumulator updated/final, output published and scratch reusable. An address-ready token cannot satisfy an activation dependency. For a down projection, one chunk can start a continued update only if its selected body exposes a legal accumulator continuation; full output still needs all K chunks.

## 7. Target body and physical plan

The backend returns `BodyTacticSpec` alternatives, each declaring semantics/numerics, feature/toolchain guards, shape/layout/alignment/tail guards, output tile and reduction footprint, acceptable enclosing participant/block configurations, warp or target-group roles, descriptor and argument lifetime, scratch/accumulator usage, staged operations, source retirement, publication and epilogue ownership. `ATOMIC_TILE`, `PRELOADABLE`, `STREAM_REDUCTION` and `EARLY_RELEASE` capabilities are independently verified, not inferred from a C++ function name. [Kernel reuse](MEGABAKE_V3_KERNEL_REUSE.md)

A provider takes an indexed reference plus an algorithm choice and emits a bounded family of **target-qualified schedules**. Tile shape, lane/warp mapping, reduction partition, MMA/copy primitive, pipeline depth, epilogue and persistent iteration are variables; semantics, casts/effects and target feature legality are constraints. Schedule generation changes physical footprints, so the affected logical tile domains and producer relations are regenerated before admission. The provider's search key is computation/shape/target, not a model-family label. A tuned tactic remains one candidate in whole-entry search, since its isolated optimum may be unusable in the mixed resource envelope.

```text
TargetExecutionPlan {
  logical_plan_hash, backend_id, target_profile_key,
  selected_body_tactics_and_versions, target_tile_refinement,
  physical_layouts_and_preparation, target_participants_and_roles,
  storage_spaces_buffers_and_lifetimes, async_operations_and_tokens,
  concrete_event_protocols, physical_worker_program_or_queue,
  joins_and_progress_argument, entry_and_binding_ABI,
  compiled_resource_report, admission_report, measurements
}
```

Lifecycle states are `lowered`, `compiled`, `admitted`, and `measured`. Compilation may change resource facts; the immutable candidate is revised or rejected rather than silently treated as legal. CUDA targets must verify the actual linked/generated entry's block compatibility, register/shared/local/code footprint and cooperative residency. A future TPU adapter supplies distinct TensorCore/VMEM/DMA/semaphore/mesh contracts. No backend may rewrite unsupported semantics merely to make a body fit.

A `StrictGridPlan` contains only device-callable work and one owned CUDA compute grid. An `ExternalPlan` may retain host-launched cuBLAS/cuBLASLt or other library kernels and a CUDA Graph; it is measured and reported separately. Both originate from the same semantic program and numerical/state contract.

## 8. Verification ladder

1. **Capture/reference:** full-step signature, lifted bindings, state effects and original versus normalized reference.
2. **Indexed semantics:** exact domains/maps, casts/masks, aliases, output/effect coverage and guarded equivalence for each algorithm choice.
3. **Logical work:** each output has one writer or complete reduction; exact producer relations, bounds, state order, address/data readiness and partial-order lifetimes.
4. **Body/target:** every logical obligation has a concrete tactic and target mechanism; participants, descriptors, stages, scratch, casts and publication match the declared contract.
5. **Progress/admission:** worker/resource wait-for graph, initialized event generations, async retirement, uniform required collectives and compiled-entry cooperative residency.
6. **Device result:** output, cache/state and repeated-call correctness under declared tolerances on the selected target.
7. **Performance claim:** whole-invocation comparison with matched best baseline, held-out timing samples and separate strict/fallback status.

A CPU reference interpreter and tiny model checker can expose logical counterexamples but cannot establish CUDA memory ordering or speed. A compilation-only run cannot establish device correctness. Numerical tolerance is chosen before timing by operation and dtype; a blanket cosine-similarity threshold is insufficient.

## 9. Reuse boundary

Reuse PyTorch for FX/export capture, reference execution, selected normalization and fallback. Reuse CUDA device libraries for target bodies when their interfaces and license/toolchain fit. Own the indexed semantic contracts, guarded algorithm choices, exact task dependence, target-plan verifier and selected persistent entry. The compiler's new work is the bridge from arbitrary supported FX arrangements to good device-callable tasks and a profitable whole-step physical program.

## 10. Concrete records an implementer must be able to construct

The schemas above are conceptual, but the first implementation needs a fixed minimum rather than a collection of optional dictionaries. The exact Python class names may differ; the following fields and invariants may not.

| Record | Minimum fields | Reject if absent |
|---|---|---|
| `CapturePackage` | source graph/reference, flat input and output pytree specs, lifted binding table, state/effect table, symbolic constraints, workload/numerical contracts, versions | Unbound parameter, missing state output, unknown graph fragment boundary |
| `ValueDesc` | internal value ID, original FX origin set/output path, shape/stride/dtype/device, storage offset, alias set, role, mutability, producer/consumers, cast origin | Unknown alias used for an in-place write or missing live output |
| `IndexedOp` | op kind, iteration and reduction domains, each operand's index expression, output index expression, expression DAG with casts, predicate, effects, reference origin | An `Opaque` node claimed as strict-supported |
| `AlgorithmChoice` | origin region, semantic guard, numerical guard, indexed reference relation/expansion, preparation actions, optional stage/epilogue contract | Pattern name without executable equivalence relation |
| `LogicalTaskFamily` | guarded tile domain, exact read/write/reduction regions, effect predecessors, output owner/finalizer, reference computation | One result with missing or competing writers |
| `BodyTacticSpec` | target guards, logical tile ABI, output and K footprint, participant/block contract, scratch/descriptor/async lifetimes, epilogue, source version | Standalone launchable kernel passed off as a callable body |
| `TargetExecutionPlan` | selected tactic versions, physical worker/grid/block, storage assignment, event protocol, progress argument, binding ABI, compiled resources, admission state | Target plan marked admitted before compiling the exact entry |

Use stable **origin IDs** derived from the original capture plus original node ordinal and output path. Internal value IDs may change under normalization, but each has an origin set; a replacement pass provides an explicit old-to-new relation. Never hash tensor addresses, runtime pointer values or Python object IDs as semantic identity. Hash graph/operator versions, guards, numerical policy, binding roles and algorithm parameters. Store source and compiled artifact hashes separately.

The indexed expression grammar initially needs integer constants, iteration/reduction variables, `+`, constant multiply, floor-div/mod for blocked layouts, bounded `select`, comparison/predicate, and a separate guarded indirect-index load for gathers. An index expression is not arbitrary executable Python. If an index cannot be bounded from guards or a runtime bounds check, strict lowering rejects it. Domain maps express values read/written; body layout maps describe where those values sit in registers/shared/global memory and belong to the target layer.

## 11. FX origin, numerical and effect proof obligations

The canonical reference for a transformed region is an **executable local FX subgraph or equivalent isolated reference evaluator** with its boundary values and effects. The checked-in `ReferenceRegion.reference` currently points to the whole-program callable; that is useful as an oracle but cannot alone prove a proposed local QKV or norm rewrite. Preserve both the whole-program oracle and local origin map. Each selected alternative needs a coverage certificate:

```text
origin nodes and outputs covered; original live boundaries preserved
old state read region; new state write region; alias and effect order
shape/stride/alignment/target guards actually checked at invocation
numerical operations whose association/cast order changes
reference expansion and comparative fixture evidence
```

Treat an FX effect as observable even when its result has no users. A normalized in-place cache write can become an old/new functional state pair, but its ordering relative to all consumers remains. An allocation may be reused only after the last ordinary and asynchronous reader retires. A `view` is a change of indexing and aliasing, not an instruction to copy; a `reshape` that materializes cannot be recorded as an alias. Missing facts remain `UNKNOWN` and either force a conservative body/layout or a diagnostic.

For an `addmm` expression, preserve `beta * input + alpha * (mat1 @ mat2)` and the point at which each multiply, accumulation and output cast happens under the captured reference. The notation `Y^T = W X^T` changes the legal algorithm's orientation, not the reference's alpha/beta or rounding contract. Split-K and online softmax are guarded numerical algorithms: they change reduction order and may require a declared tolerance and edge-case checks. No generic reassociation flag silently applies to every op.

## 12. End-to-end tiny derivation

Use `LINEAR_TINY` with `X[1,33]`, `W[17,33]` and an output tensor `Y[1,17]`. A captured `aten.mm` plus `aten.add` or `aten.addmm` may spell the projection differently. Normalize only after preserving whether bias, alpha/beta and a cast were present. The indexed contraction has domain `b in [0,1), n in [0,17), k in [0,33)`; `X` map `(b,k)`, `W` map `(n,k)`, output map `(b,n)`, a K reduction and an explicit cast expression. A transposed weight view instead changes the W map and strides; it is not recognized by tensor shape alone.

A conservative tile width eight yields three logical output tiles: `n=0..7`, `8..15`, and `16` with an inactive-lane predicate. The producer relation for a downstream pointwise tile is only the projection tile whose output region it reads. A cache write is an effect task with a bounded destination and is not removed when its result is unused. A K-parallel SIMT body may own an entire output tile; a split-K candidate creates multiple reduction contributors and a finalizer before downstream publication. Both realizations refer to the same indexed output/effect meaning.

A target may choose a different tile width or the output-channel-major tensor-core algorithm. That changes output/K footprints and forces regeneration of task domains/dependence. It does not rewrite original FX, erase the unfused reference, or allow model-name rules. The [implementation handbook](MEGABAKE_V3_IMPLEMENTATION.md#9-worked-vertical-trace-that-every-agent-can-follow) specifies the corresponding cards and rejection cases.

## 13. Migration from the currently checked-in semantic frontend

`frontend/capture.py`, `normalize.py`, `facts.py` and `effects.py` are useful starting points. `frontend/semantic.py` currently selects named `SemanticNode`s and leaves unmatched nodes as `ReferenceRegion`s; `composites.py` enumerates overlaps over that named cover; `layers.py` computes fingerprints. The migration order is:

1. Add origin-stable `IndexedTensorProgram` alongside `SemanticGraph`; keep old CPU tests as a regression oracle.
2. Give every declared supported primitive a local reference and a generic body route. A leftover `ReferenceRegion` is `unsupported_for_strict` until it has one.
3. Re-express `match_linear`, `match_norm`, `match_attention`, `match_rope`, `match_state` and composite candidates as guarded `AlgorithmChoice` proposals over indexed origins, retaining their useful recognition tests.
4. Strengthen cover verification from “no duplicate named ops” to **complete original live FX output/effect coverage**.
5. Promote `LayerSummary` to `RepeatRegion` only after flat expansion verifies per-iteration weights, carried values/state and exceptional variants.

A card may temporarily expose both old and new records. Only the new indexed/logical path may assert V3R strict coverage. Keep versioned adapters at the FX boundary; later Inductor buffer or external-kernel objects cannot be reverse-inferred as exact high-level semantics. The [current code inventory](MEGABAKE_V3_IMPLEMENTATION.md#2-what-the-current-repository-gives-us) identifies the relevant files.

## 14. First strict cached-step ABI and public compiler surface

The existing `WorkloadSpec` fixes benchmark intent, and `NumericalPolicy` fixes allowed arithmetic differences; neither by itself says which FX placeholder is old KV state or which output becomes next state. V3R-001 therefore adds a versioned **`StepABI`** record for strict compilation. Its required information is:

```text
StepABI/v1 {
  ordered_user_inputs: placeholder IDs and pytree paths,
  lifted_bindings: placeholder -> parameter/buffer/constant identity and lifetime,
  old_state_inputs: placeholder IDs, layout, alias set and capacity,
  state_effects: exact write/read regions and order,
  new_state_outputs: output paths or declared in-place alias transitions,
  user_output_tree: full pytree spec and ownership/lifetime of each leaf,
  position_and_valid_length: source IDs, old L, append index L, new L+1,
  batch_rule: uniform valid length for the first supported slice,
  invocation_preparation: packing/copies/descriptors and amortization policy,
  guard_set: shape, stride, dtype, capacity, feature and numerical guards
}
```

For the first functional cache mode, `run` receives old state, returns new state, and leaves old state unchanged unless the captured reference explicitly permits aliasing. An in-place mode is a separate ABI version/guard, not an implementation detail. The cache convention is `0 <= L < C`, append new K/V at `L`, attend to the declared valid range after that write, and report `L+1`; a different model convention needs a new explicit mapping. Output logits retain the captured output tree and dtype. The optional HF helper may adapt user-facing shapes, but **the same adapter** wraps reference and all baselines.

The first public interface lives at `megabake.v3.compile_fx`, leaving the checked-in `megabake.compile_fx` legacy behavior intact until an explicit migration. Its semantic signature is:

```text
megabake.v3.compile_fx(graph_or_export, *, example_inputs, workload, numerical_policy,
           step_abi, target=None, mode="strict_only") -> CompiledStep
CompiledStep.run(user_inputs, old_state, *, output_buffers=None)
    -> (user_outputs_in_declared_tree, new_state, ExecutionRecord)
```

`mode="strict_only"` either returns an admitted one-grid artifact or raises an origin/guard/target diagnostic. `mode="allow_external"` may select a vendor-preserving plan but `ExecutionRecord.plan_class` must say `ExternalPlan` and include its grid count; it cannot satisfy a strict claim. A guard failure at runtime either selects a separately compiled valid specialization or returns the diagnostic/fallback allowed by the requested mode. Binding new token content, descriptors or cache positions is part of invocation work and has a stated cost. A compiled artifact cannot hold a borrowed output buffer past its declared lifetime.

`ExecutionRecord` is semantic reporting: an implementation may keep immutable per-specialization metadata on `CompiledStep` and return a lightweight view, avoiding a fresh heavyweight Python allocation in every timed step. Any reporting overhead actually present in the callable remains in the complete-call benchmark; diagnostic profiling can use a separate path.

The implementation may choose dataclasses or an existing typed-record style, but it must preserve these semantics and serialize/hash all compile-relevant fields except raw pointer values. [V3R-001](MEGABAKE_V3_IMPLEMENTATION.md) owns the record; V3R-028/038 own the runtime and public surface.
