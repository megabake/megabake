# MegaBake V3: IR and reuse contracts

Status: pipeline-first proposed design, 2026-09-09. The [architecture](MEGABAKE_V3_ARCHITECTURE.md) owns scope.
Examples below are schematic contracts, not implemented Python APIs.

## 1. When a representation earns its existence

An IR is justified when a transformation needs information or invariants that the preceding
representation cannot conveniently express. A new name, serialization format, or list of model
layers is insufficient. Conversely, a dialect within FX can be a meaningful IR without requiring
a new graph container.

| Stage | Information added | Transformations enabled |
|---|---|---|
| Normalized FX | Tensor, shape, alias, numerical and effect facts | Safe simplification, guards, decomposition, reference execution |
| SemanticGraph | Defined FatOps and their reference regions | Semantic matching, alternative composites, implementation selection |
| ExecutionPlan | Body stages, tile ownership/footprints, reduction continuations, movement and readiness/release tokens | Joint fusion/tiling/scheduling, bounded overlap, safe storage reuse, CUDA generation |

`TensorFacts`, `LayerSummary`, and `TargetProfile` are side analyses. A memory allocation table is
a field of the execution plan. Candidate and final plans have the same schema, with unresolved
choices prohibited in a final plan. This avoids keeping multiple subtly inconsistent graphs alive.

## 2. The actual Inductor handoff

The inspected version is PyTorch 2.6.0, matching the historical GPU environment. Its
[compile flow](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/compile_fx.py) performs
FX processing before GraphLowering. The latter eventually constructs a Scheduler over lowered
operations in [graph.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/graph.py).
By then operations include loop/buffer/external-call representations, not just ATen nodes.

The [post-grad pass pipeline](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/fx_passes/post_grad.py)
contains pattern/fusion work and later mutation-oriented processing. Consequently:

- There is no clean promise of “all optimizations, no fusion.”
- A custom pass hook transforms a graph; it is not itself a supported alternate-backend exit.
- Taking over immediately before Scheduler would inherit substantial lowering decisions.
- Reconstructing ATen from that representation is avoidable work and can lose intent.

V3 recommendation: own a backend/export adapter that receives functional FX, runs an explicitly
selected normalization sequence, then exits before GraphLowering. Reuse early Inductor passes only
through a small version-pinned adapter. Do not monkeypatch the entire compiler pipeline.

Use [ExportedProgram](https://docs.pytorch.org/docs/2.6/export.html) when available; preserve its
graph signature and constraints through decomposition. A bare GraphModule needs explicit example
input/spec information and a policy for state and mutation. It cannot silently acquire the same
guarantees as an exported program.

The existing [adapter](src/megabake/schedule_compiler/inductor_passes.py) already demonstrates selected
decompositions and a private early-pass call. It is a starting point, not a production stability
contract. Audit whether each pass mutates its argument or returns a replacement; propagate the
replacement and metadata intentionally. Keep original and normalized reference execution tests.

### Proposed pass order

1. Validate input/output/state signature, supported devices and inference assumptions.
2. Functionalize supported effects; retain observable state explicitly.
3. Apply selected decompositions, preserving recognizable high-level operations where possible.
4. Propagate fake tensor/shape metadata; collect alias and numerical facts.
5. Run allowlisted simplifications, constant folding and effect-aware DCE.
6. Recognize FatOps using semantic guards; retain reference expansions.
7. Enumerate bounded composites without deleting alternative implementations.

The allowlist must be tested on the pinned PyTorch version. “Reuse Inductor” does not mean import
every private pass or duplicate its entire optimizer. Inductor scheduling, Triton fusion and
autotuning remain available in the fallback baseline; they are not automatically inherited by
the persistent backend.

## 3. Normalized FX and TensorFacts

For each value, record:

```text
TensorFacts {
  logical_shape, symbolic_constraints, dtype, device,
  strides, storage_offset, alignment_if_proven,
  alias_set, mutability, layout_constraints,
  producer, consumers, provenance
}
```

Graph-level facts include parameters/constants, user inputs, returned outputs, state inputs and
updates, permitted numerical policy, and guards. Facts are proven or unknown, never guessed from
the model name. Contiguous weights do not prove contiguous activations. Unknown alignment selects
a safe implementation or a checked specialization.

Views are not necessarily copies. Zero-stride broadcast, transposes, slices and aliases must survive
normalization. A view can disappear only when the selected consumer represents its indexing exactly.
Repacking invariant weights may happen during session creation, with memory and setup cost reported.
Repacking a changed activation is runtime work.

A strict entry rejects unsupported effects, cross-device operations, unbounded data-dependent
control flow, or incompatible custom operations. The diagnostic names the node and missing
contract. The normal callable may fall back, visibly. The compiler must not mislabel partial
recognition as arbitrary-FX single-grid support.

## 4. FatOps: semantics first, algorithms later

The useful aspect of the [vLLM IR proposal](https://github.com/vllm-project/vllm/issues/32358) is a
functional operation dialect with reference behavior and late implementation selection within
the PyTorch compilation ecosystem. V3 adapts that principle, not a promise that the RFC supplies
a complete persistent compiler or an existing stable MegaBake API.

Each FatOp definition requires:

```text
name + semantic_version
input/output/state signature
reference expansion
shape/dtype/layout rules
alias/effect contract
numerical policy and permitted implementation differences
recognition guards + origin FX nodes
```

Each implementation is a separate record with support guards and a lowering contract. Semantic
versions change when behavior changes, not whenever a faster kernel is added.

| FatOp | Essential distinctions to retain |
|---|---|
| Linear | Transpose/stride, bias, accumulation policy, output cast, quantization absent/present |
| RMSNorm | Reduction axes, epsilon placement, accumulation/cast order, weight versus `1 + weight` |
| Pointwise | Exact expression DAG and intermediate casts; not an arbitrary opaque callback |
| SwiGLU | Which input is activated, SiLU definition, cast/rounding boundaries |
| RoPE | Pairing convention, rotary dimension, positions, frequency/scaling rule, dtype |
| SDPA | Scale, mask, causal alignment, window, head mapping, dropout, output policy |
| Cache/state update | Indexing, old/new state, capacity, valid lengths, alias/effect ordering |

Example: a plausible RMSNorm reference is `cast(x * rsqrt(mean(float(x)^2) + eps)) * weight`.
Another model multiplies by `1 + weight`, or performs the final multiplication before the cast.
Those are not interchangeable just because all are called RMSNorm. Record the actual expression.

Likewise, `SwiGLU(g,u) = silu(g) * u` does not inherently include either projection. The reference
expansion determines where FP16/BF16 rounding occurs. Fusing linears with the gate is an
implementation alternative, not permission to change those boundaries silently.

The pipeline template is more general than SwiGLU: `phi(gate(x)) * up(x)` can retain its exact
Pointwise expression when phi is not SiLU. For example, the inspected
[Gemma MLP](https://raw.githubusercontent.com/huggingface/transformers/v4.50.0/src/transformers/models/gemma/modeling_gemma.py)
selects its activation from configuration. Check the captured graph and chosen checkpoint/version;
never replace a GELU-gated region with SiLU because a scheduling template is named after Llama.

Reference expansion can be an explicit graph region rather than a hand-rewritten formula. Retaining
it helps diagnose failed matches and supports differential tests. Matching uses graph structure,
constants, axes and facts; names/configuration are hints only.

### Bounded composites

Recognize overlapping candidates such as:

```text
RMSNorm -> QKV projections
gate projection + up projection -> SwiGLU
Linear -> bias/residual/activation
RoPE -> cache write
```

Use bounded beam/template selection jointly with tile shape, body capability, transport and
schedule. Do not greedily finalize the semantic cover before evaluating consumer readiness.
Require complete coverage, no duplicated effects, and retained live boundary values in the final
plan. Allow duplication only for explicitly pure, costed recomputation. Preserve an unfused option.

A fused semantic region may still lower to several tile actions. Conversely, several semantic
nodes can share one CTA-local implementation. Semantic, readiness and launch boundaries differ.
The final semantic cover identifies which operations are implemented, while its action graph
identifies when each portion executes. The two views must have an explicit coverage mapping.

## 5. Attention and the supplied config

The supplied [Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json),
accessed 2026-09-09, identifies Qwen3.5 architecture classes and contains a 64-layer text stack
with three `linear_attention` layers for each `full_attention` layer. It also carries recurrent
attention dimensions. The repository slug, config class and actual computation must not be
collapsed into one assumed model identity.

This is Transformers model metadata, not an executable GGUF IR. A layer list does not establish
activation formulas, masks, state lifetimes or the implementation of “linear attention.”

| Information | Semantic or implementation? | V3 location |
|---|---|---|
| Causal/full/windowed connectivity | Semantic | SDPA attributes/reference |
| GQA head mapping | Semantic | SDPA shape/head contract |
| FlashAttention algorithm | Implementation | Candidate implementation |
| Paged KV layout | State/storage interface, often with logical indexing | State facts + plan |
| Gated recurrent update | Different semantic state transition | Defined recurrent FatOp/region |
| Layer ordering from config | Hint requiring graph verification | LayerSummary |

[FlashAttention](https://arxiv.org/abs/2205.14135) computes exact attention with an IO-aware
algorithm. It is not a third mathematical attention category alongside full and linear attention.

The corresponding [Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
includes convolution state and a gated recurrent matrix update in its delta-net path. Therefore
lowering all config-labelled attention to SDPA would be incorrect. A future recurrent op must
define that recurrence, its state dtype and update order, not merely carry `kind=linear`.

Initial V3 scope is ordinary cached attention. Recurrent and multimodal regions remain reference
regions/unsupported strict paths until implemented and tested. The architecture can represent
them now without pretending to have an optimized body.

## 6. LayerSummary, and when Layer IR becomes real

A summary can record an ordered repeated region, stable structural fingerprint, layer index,
parameter bindings, state bindings and boundary values. Verify structural matches against FX.
If one layer has a different mask or norm, its fingerprint differs.

Immediate uses do not require a new semantic IR:

- Compile a reusable function/template once for repeated shapes.
- Reuse tuning results where guards and layouts match.
- Display per-layer cost and state ownership.
- Associate compatible composite candidates with repeated graph regions.

Promotion is justified if a transformation must reason about a loop carrying explicit state,
prove a cross-iteration invariant, or restructure repeated regions while preserving iteration
semantics. A structured loop then needs bounds, carried values/effects and a reference meaning.
Do not add it merely to reproduce a configuration file's hierarchy.

Code reuse is not automatic weight reuse: repeated layer shapes usually have different weights.
A template loop may reduce instruction footprint without reducing the model's weight traffic.

## 7. ExecutionPlan

```text
ExecutionPlan {
  semantic_graph_hash, specialization_guards, numerical_policy,
  target_requirements,
  semantic_cover: [origin_regions, selected_body_or_composite],
  bodies: [version, guards, thread_roles, stage_capabilities, resource_contract],
  tiles: [body, logical_domain, output_region, read_and_reduction_footprints],
  actions: [tile, kind, inputs, outputs, worker_or_cohort, required_tokens],
  reductions: [output_owner, accumulator, ordered_chunks, finalizer, cast_policy],
  edges: [producer_action, consumer_action, exact_region, transport, layout],
  events: [producers, expected_count, consumers, scope, initialization, generation],
  buffers: [size, alignment, address_space, ownership, acquire_release_actions],
  async_operations: [issuer, source, destination, completion, source_retirement],
  schedule: [regions, worker_programs, cohort_sizes, lookahead, joins, progress_proof],
  launch: [entry, block_shape, resident_worker_count, dynamic_smem],
  invocation_bindings, output_and_state_contract,
  compiled_resources, verification_report, measurement_provenance
}
```

This is a logical schema, not a requirement for one object allocation per action or a general
instruction interpreter. Repeated tiles/actions can be represented by affine domains and template
loops; finalized descriptors may be compiled away. Candidates and selected plans use this schema.

An implementation's logical tiles describe work, not resident CTAs. Each output has one writer,
unless an explicit reduction defines contributors and finalization. Body capabilities distinguish
indivisible tiles, separately preloadable tiles and continued reductions. Actions have typed
preconditions: an address-ready token cannot satisfy a data-ready dependency; scratch release
cannot satisfy output publication. See the [action contracts](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md#2-actions-and-capability-contracts).

An output tile of `down(h)` ultimately reads its entire K footprint. A `reduce_update(I,J)` reads
only `h[J]` and `W[I,J]`, plus its owner's prior accumulator. That permits early computation, not
early final output. All chunks must cover K exactly as required before `reduce_finalize`. A head
attention tile similarly depends on its precise Q/K/V/state region, not all unrelated heads.

For proven invariant weights, activation readiness need not prevent an independent preload into
a reserved slot. Preserve any real mutation/effect dependencies. Express address dependencies
separately; do not mark a next operation entirely
unready when only its activation is missing. A dynamic address dependent on an earlier value
does retain that dependency. No speculation beyond guarded memory bounds is allowed.

The primary lowering generates bounded pipelined worker/cohort programs. Barrier control is a
second schedule policy for the same plan. A general scheduler IR, machine-instruction Tile IR or
model-specific Layer IR is not necessary to express these decisions.

### Dependency scopes

| Scope | Example | Legal mechanism |
|---|---|---|
| Thread | Local expression | Program order |
| Warp | Cooperative reduction | Required warp synchronization and participation contract |
| CTA | Shared tile producer/consumer | Block barrier or documented async protocol |
| Grid | Materialized phase boundary | Cooperative grid synchronization |
| Cross-CTA tile | Published head/K chunk | Device-scope publication/acquire plus async completion and correct event lifecycle |
| Resource token | Reusing staging or accumulator storage | All prior accesses retired, correct ownership and bounded credit protocol |

Do not treat an atomic increment as a complete memory-ordering proof. The first pipeline requires
initialized events, exact producer sets, no premature zero-ready state, scoped publication,
safe reuse and a forward-progress argument. Its conservative initial protocol is defined in
[publication](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md#9-cross-worker-publication-protocol).
Worker order, resources and collective participation enter the proof, not just tensor-DAG edges.

## 8. Memory and state correctness

Allocate from the selected schedule's partial order, not the original FX order or an estimated
timeline. Incomparable uses may overlap. Async copies, continued accumulators and output retention
extend lifetimes; distinguish source-access retirement from destination visibility. If two buffers
alias, every access to the old value must retire before overwrite. Any body/tiling/pipeline change
invalidates affected allocation, event and progress analyses and requires reverification.

Static per-CTA staging slots and distinct inter-CTA activation chunks within a region are enough
initially. Reuse across a region join requires all readers/async work to finish. Inter-CTA ring
reclamation with consumer acknowledgements is not silently assumed to exist.

State can be represented functionally:

```text
(logits, kv_next) = decode_step(token, position, kv_old, weights)
```

In-place cache storage is a lowering decision. Its proof includes valid indices, capacity, read/write
ordering, no destructive alias to still-live data and explicit ownership across invocations. DCE
cannot remove a required state update simply because the user ignores a returned tensor. A session
API may designate state updates as required effects independently of the visible logits result.

Return ownership also matters. Reusing the same output allocation on the next call changes the
normal tensor lifetime contract. Use owned outputs by default or a separately documented borrowed
buffer/`run_into` contract. Compare equivalent baselines.

## 9. Verification ladder

1. **Reference equivalence:** run original, normalized and FatOp-expanded graphs on shared inputs.
2. **Plan structure:** semantic/tile coverage, exact footprints, unique writers/reduction coverage,
   typed tokens, effect order, bounds and guards.
3. **Storage:** alias legality, partial-order lifetimes, async completion versus source retirement,
   event initialization, accumulator lifetime and session ownership.
4. **Participation:** every required barrier reached uniformly; every body receives its legal
   threads, launch shape and scratch; cooperative residency checked on the compiled entry.
5. **Progress/publication:** augmented worker/resource wait-for graph, correct atomics/scopes,
   no stale generations, all producers runnable; tiny interleaving tests and device litmus tests.
6. **Device differential tests:** values and state, including repeated calls and bucket boundaries.
7. **End-to-end numerical tests:** agreed reference policy, adversarial ranges, long-context state
   and multi-step divergence. A large model-to-model max error alone proves little.

For the pipelined executor, use induction over completed semantic actions: acquired dependencies
establish valid inputs; a legal body/update preserves the output or accumulator invariant; proper
publication makes completed values available; access retirement permits storage reuse. Finalizers
cover all required contributions. Separately prove progress under the admitted execution model.
The barrier control has the simpler phase-induction special case. These are proof obligations,
not a claim that either generated implementation has been formally verified.

Numerical tolerances are chosen before timing by dtype and operation behavior. Exact integer,
index and shape semantics remain exact. Floating-point allowances must be explicit; a blanket
cosine-similarity threshold or “torch.compile also differed” is not a sufficient correctness gate.

## 10. Reuse boundaries

Reuse PyTorch for capture, reference semantics, metadata, selected normalization and fallback.
Reuse existing device libraries for supported math below their host-launch layer. Reuse known
dependency/scheduling ideas only with their correctness contracts. Own the FatOp contracts,
execution plan, composition verifier and specialized persistent entry.

This concentrates new compiler work where it changes the result: selecting and composing good
implementations without losing semantics. It avoids rebuilding a tensor frontend, a universal
hardware-description language, or another general CUDA math library.
