# MegaBake V3: optional pipelines, scheduling and progress

Status: normative proposed execution contract, revised 2026-09-27. The [architecture](MEGABAKE_V3_ARCHITECTURE.md) makes body quality and complete-step latency primary; this file describes legal scheduling candidates and their proof obligations. None is assumed profitable without target measurement.

## 1. Legal work before a scheduler

The `LogicalExecutionPlan` provides parametric task domains, indexed read/write/reduction maps, exact producer relations, effects and lifetime constraints. It does not assign CUDA CTAs or prescribe a queue, cohort, staging depth or barrier. A backend may refine tile domains after choosing a body, then regenerate and reverify dependencies. This permits a static CUDA worker program, a dynamic expert dispatcher or a future gridless TPU schedule from the same semantics.

One complete operator is often too coarse: a Q head can be ready before unrelated QKV output; one gated hidden chunk can be ready before the rest. A machine instruction is too fine for the common plan. The useful unit is a **logical tile with a body-supported stage contract**. A body may expose only an indivisible tile. No planner may invent preload or continued-reduction stages inside an opaque body.

Luminal's [symbolic block domains and barrier strides](https://blog.luminal.com/p/compiling-models-to-megakernels) show how to describe many tasks compactly. MPK's [event fusion and linearization](https://arxiv.org/html/2512.22219v2) show why materializing one event per producer/consumer pair is wasteful. V3 derives exact relations symbolically and lowers only the selected representation to runtime metadata.

## 2. Typed actions and completion meanings

| Action | Required condition | Completion |
|---|---|---|
| `reserve(slot)` | Prior accesses retired; capacity and ownership legal | Exclusive staging lease |
| `preload(tile, slot)` | Source address/data known; descriptor valid; slot reserved | Copy issued, then load complete |
| `compute(tile)` | Required input data acquired; participants/body scratch legal | Declared output region computed |
| `reduce_begin(owner)` | Accumulator storage available | Initialized continuation |
| `reduce_update(owner, chunk)` | Chunk complete and acquired; prior update ordered | Updated private accumulator |
| `reduce_finalize(owner)` | Every required chunk incorporated once | Final cast/epilogue and complete output |
| `publish(region)` | Required writers and async stores complete at target scope | Cross-worker data-ready token |
| `release(slot)` | Every source access and reader has retired | Slot may be overwritten |
| `join(region)` | Required worker work, reads and async operations completed | Coarse safe boundary |

The plan must distinguish **address known**, **input data ready**, **load issued**, **source read retired**, **destination visible**, **MMA complete**, **output published**, **reduction final**, and **storage reusable**. One cannot substitute for another. For example, a next-layer weight address may be known before the next activation, permitting an independent load into a free slot. That does not permit the next GEMM to run before the activation arrives.

Async mechanisms have target-specific completion semantics. An atomic counter update is not by itself a proof that TMA writes, register-to-shared exchanges or global stores are visible to a consumer. The CUDA adapter supplies the required instruction waits, proxy fences and acquire/release scopes. The common logical plan records the needed visibility and lifetime; it does not name a CUDA primitive. [PTX async bulk wait documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)

## 3. Candidate scheduler policies

| CUDA policy | Appropriate first use | Price to measure |
|---|---|---|
| Barrier control | Correct simple baseline with the same bodies | Grid joins and activation handoffs |
| Static resident-worker programs | Predictable dense decode | Assignment imbalance, tile tails and program/code size |
| Bounded pipelined programs | A body exposes useful early work | Extra events, live scratch, bandwidth contention, occupancy loss |
| Dynamic or hybrid dispatch | Data-dependent expert work or measured stragglers | Queue/atomics, scheduler participants and metadata traffic |

For the initial dense decode path, generate a static worker program and a matched barrier control. Head-ready work, streamed MLP and weight lookahead are **optional alternatives**, not mandatory features of the first winning entry. A global queue is introduced when the target workload and measured imbalance justify its overhead. MPK combines ahead-of-time and just-in-time dispatch; Luminal describes a global queue. Their results justify candidates, not a universal policy. [MPK](https://arxiv.org/html/2512.22219v2), [Luminal](https://blog.luminal.com/p/compiling-models-to-megakernels)

On CUDA, a cooperative grid contains `P` resident worker CTAs admitted from the **actual compiled entry** and selected block/dynamic-shared-memory configuration. Logical tile count can be thousands even if P is small. `blockIdx.x` is a worker identifier, not a guarantee of physical SM placement. All workers reach any retained grid join, even if a worker has no arithmetic in that phase. A different block shape can make a body or the whole grid illegal; this is a candidate rejection, not a runtime surprise.

## 4. Head-ready QKV and decode attention

A producer tile may complete Q, K or V for one head/GQA group. The attention consumer needs its exact Q, mapped K/V, positional transform, valid cache length, mask and any current-token cache slot. It can start after those inputs are visible; unrelated head groups do not impose a dependency.

```text
project group 0 -> RoPE/cache publication -> attention group 0
project group 1 -> RoPE/cache publication -> attention group 1
                    ...
all required attention outputs -> output projection or legal continuation
```

A packed QKV tensor-core body may have better throughput but publish heads later than separate or smaller projection bodies. The search keeps both, and its cost includes changed weight layout, duplicate activation loads and whole-entry resources. A current-token cache write must be complete and visible before dependent attention reads. GQA mapping and causal/window masks follow the indexed semantics, not a template name.

Online-softmax decode attention maintains running maximum, normalizer and weighted-value accumulator. Split-context work needs a correct combine of partial maxima/normalizers/accumulators and explicit state/bounds. Empty or fully masked rows follow the reference; never evaluate undefined `-inf - -inf` and call it equivalent. A strong external attention kernel remains a measured `ExternalPlan` control.

## 5. Gate/up/down choices

For a gated MLP with the exact reference activation and casts:

```text
g = W_gate x; u = W_up x; h = phi(g) * u; y = W_down h
```

Compare three target plans:

1. Materialize a complete `h`, then run the best full-K down body.
2. Publish matched hidden chunks and let an output owner perform ordered FP32 continued reduction updates.
3. Use independent partials and an explicit combine/finalizer when parallelism demands it.

The continued version is legal only if its down body exposes `reduce_begin/update/finalize` with a supported accumulator, cast order and participant scope. Every output needs all required K chunks before final cast/residual. A cuBLASDx internal K pipeline is not automatically this interface. Long-lived accumulators can raise the entire entry's register envelope and destroy the expected overlap. A full-K tensor-core body may win despite the later start.

Initially, distinct cross-CTA hidden chunks may live in global addresses for a region and be reused after a proven join. Global-address-space traffic may hit L2; it is not automatically HBM traffic. A producer on another CTA cannot forward registers or CTA shared memory without an explicit supported cluster protocol. Fuse gate/up/gating locally only where output ownership and numerical boundaries allow it.

## 6. Cross-task and cross-layer weight movement

Stable weight addresses can be known before the next activation. A target plan may reserve a staging slot and issue a transfer early:

```text
known weight address + free slot -> issue load -> load complete ---+
current work -> next activation published and acquired ------------+-> compute
```

This can cross tile, operation or layer boundaries in a verified `RepeatRegion`. It does not imply keeping an entire next matrix in CUDA shared memory. A bounded zero/one/two-slot menu is a useful initial CUDA search, with more depth only when measured. Price extra shared storage, descriptor setup, occupancy, HBM/L2 contention, issue bandwidth and fill/drain effects. Two overlapping HBM-bound stages cannot each be credited with full standalone bandwidth.

Inferact's [TPU schedule](https://inferact.ai/blog/tpu-megakernels) uses large software-managed VMEM and async DMA across layer boundaries. The logical opportunity transfers; the physical lifetime, capacity and schedule do not. Hazy's [B200 breakdown](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) reports activation handoff cost far above its remaining weight-wait cost, so the CUDA search must also consider activation ownership and publication, not only prefetch depth.

## 7. Storage allocation and buffer reuse

Derive logical conflicts from the selected task partial order, not FX topological order or estimated timestamps. An allocation may be overlaid only if every read and async source access to the old value retires before the new writer. An output that the caller retains cannot be silently overlaid on next-call scratch. Backend allocation respects space reachability, alignment, descriptor lifetime and coexisting preload/current/epilogue stages.

For early CUDA plans, static per-CTA staging slots and distinct inter-CTA activation addresses within a region are sufficient. Reuse after a verified join is simpler than an inter-CTA credit ring. Finer reclamation may be added when measured storage pressure warrants it, with explicit consumer acknowledgements and generation handling. Shared scratch time-multiplexing avoids summing mutually exclusive allocations; peak entry reservation and register allocation still constrain occupancy.

## 8. Event initialization, publication and progress

For static producer sets, initialize each event's generation and exact expected producer count before any consumer can observe it. A consumer must never interpret an unregistered zero counter as ready. A uniform initialization/join phase is a conservative first CUDA protocol. For dynamic task counts, registration itself has a safe close/epoch rule; a count derived from routed work cannot be published before the routing decision and all corresponding producers are registered.

A producer publishes only after its output region is complete and visible at the consumer scope. Acquire occurs before consuming. Reused event storage needs generation separation to prevent stale success. Body-provided async stages transfer explicit outstanding-operation tokens to the enclosing worker program. Source retirement and output publication remain separate.

Progress proof examines the physical worker/resource wait-for graph:

- Every waiting consumer has a producer able to run within the admitted resident worker set.
- No worker holds the only buffer or participant credit needed by its own dependency producer.
- All threads/warps required by a body collective take the same legal branch and reach its waits.
- All grid joins have uniform worker participation and no earlier cohort-specific wait cycle.
- Dynamic queues cannot strand ready tasks behind an unregistered or never-triggered event.

An acyclic tensor DAG alone does not prove any of these. The target verifier supplies target-specific memory ordering and forward-progress rules; CPU interleaving simulations expose logical counterexamples but do not certify device behavior.

## 9. Selection and evidence

The objective is complete invocation time under exact semantics, legal resources and the best matched baseline. Estimate a constrained critical path with joint contention; keep unknown costs marked unknown. Compile bounded finalists and inspect actual resources. Measure a barrier control with the same body mix, then mechanism-distinct candidates: fusion, head readiness, streamed MLP, weight lookahead and any dynamic dispatch. Reject individual mechanisms when their net latency loses; a valid one-grid win does not have to use all of them.

Record stage traces, tile counts, tail waves, activation handoffs, physical-byte counters where valid, event costs, spills and code size in diagnostic runs. The [performance model](MEGABAKE_V3_PERFORMANCE_MODEL.md) defines claim and sampling rules. A missing trace does not become proof of overlap from a plausible timing diagram.

## 10. First safe schedule an agent should implement

The first CUDA program is deliberately simple. After compiling and admitting the exact selected entry, launch `P` resident cooperative worker CTAs. Initialize event generations and all expected producer counts in a uniform phase. Assign finite logical task ranges to workers; each worker computes its assigned tiles, drains its body's declared async operations, publishes results, and reaches every required grid join. A worker with no arithmetic in a phase still joins. Only after the last needed producer completes may a dependent phase read global results. All workers terminate. This schedule may materialize activations and lose performance, but it gives a correct same-body control for later overlap variants.

The physical planner must either prove that every wait's producer can run among the admitted workers or use a uniform join that cannot strand producers. Logical acyclicity does not imply this: if all P workers wait for tasks assigned to workers that are not resident, progress fails. `blockIdx.x` is a worker ID, not a physical SM ID; the entry cannot rely on per-SM pinning. A body collective requiring every thread/warp in its scope must be invoked uniformly by those participants even when a logical tile has no active output lanes.

## 11. Concrete event and buffer state machines

For a static producer set, one event generation has these conceptual states:

```text
UNINITIALIZED
  -> REGISTERED(epoch, expected_producers > 0)
  -> PRODUCING(epoch, remaining = expected_producers)
  -> READY(epoch, remaining = 0, payload visible)
  -> RETIRED(epoch, all consumers finished)
  -> REGISTERED(epoch+1, ...)
```

Registration must become visible before any consumer checks readiness. Each producer publishes its payload with target-correct release semantics **after** all of its async stores and reduction contributions complete, then decrements exactly once. A consumer observes the matching epoch/zero state with acquire semantics before reading. Reuse requires all old consumers and async reads to retire; a previous token's zero counter cannot satisfy the next token's wait. For an empty producer set, a separate explicitly initialized ready state is needed; do not treat an unregistered zero as ready. Dynamic producer counts require a close/registration phase after routing; do not decrement toward zero while producers are still being discovered.

A staging slot has `FREE -> RESERVED -> LOAD_PENDING -> LOAD_VISIBLE -> IN_USE -> LAST_READER_RETIRED -> FREE`. The source buffer has its own `SOURCE_READ_PENDING -> SOURCE_READ_RETIRED`; a destination load-complete event is not always proof that every source read has retired on every mechanism. The CUDA provider maps these logical transitions to its actual TMA, `cp.async`, MMA, proxy-fence and barrier contracts. Unsupported or unproven transitions reject that body/stage plan. This specification intentionally does not prescribe one PTX fence for all SMs.

For a QKV head group, the projection task must finish the complete K reduction and required cast before publishing Q/K/V. If RoPE and current-token cache append are separate tasks, attention waits for their exact published outputs and valid length. A packed QKV body that only exposes whole-projection completion cannot claim earlier head publication. For a streamed down projection, the owner accepts each hidden chunk exactly once; `reduce_finalize` waits for the full declared chunk set, then casts/stores/publishes the output. [IR task maps](MEGABAKE_V3_IR_AND_REUSE_PLAN.md#6-parametric-logical-task-plan) define the producer relation; body stage metadata defines what can actually be published.

## 12. Required proof and ablation record for an optional pipeline

Each optional head-ready, streamed-MLP, prefetch or dynamic-queue candidate records: the unchanged FX/indexed reference; changed body/tile/layout; event and buffer state transitions; initialized producer set and epoch; participant/collective scopes; wait-for graph; compiled entry resources/residency; diagnostic trace showing the claimed early work; same-body barrier/zero-lookahead control; and unprofiled complete-call samples. A trace that merely overlaps colored bars does not establish useful overlap if body quality or HBM contention worsens. A candidate may be correct and slower; retain the evidence and select the faster legal plan.

| Counterexample | Verifier response |
|---|---|
| Consumer tests zero before producer registration | Reject event protocol |
| One CTA waits while its required producer cannot be resident | Reject progress plan |
| Body returns with outstanding TMA source read and slot is overwritten | Reject storage lifetime |
| Partial QKV K tile publishes an attention-ready head | Reject readiness contract |
| `reduce_finalize` runs after some, not all, hidden chunks | Reject reduction ownership |
| Fixed `blockDim` prevents a selected body collective from participating uniformly | Reject body mixture |
