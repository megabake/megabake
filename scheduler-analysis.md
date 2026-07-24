# Scheduler Analysis: BSP vs Event-Driven

## BSP (Bulk Synchronous Parallel)

**How it works:** All SMs execute the same task → `grid.sync()` → next task. Dead simple — the scheduler is ~20 lines in `megakernel.cu`.

**Pros:**
- Trivial to implement and debug — no race conditions, no deadlocks, deterministic execution order
- No atomic overhead — zero contention on shared state
- Predictable memory access patterns — all SMs read/write the same buffers simultaneously, good for L2 cache hit rates
- CUDA Device LTO optimizes register allocation globally because the control flow is simple
- Easy to profile — each task is a clean, measurable unit

**Cons (perf):**
- **Barrier cost is fixed per task regardless of task size.** A 0.5us elementwise op pays the same ~5-10us `grid.sync()` as a 500us matmul. For 50-80 tasks per forward pass, that's 250-500us of pure waste.
- **Slowest SM dictates wall-clock per task.** If attention tiles unevenly (8 tiles on 32 SMs), 24 SMs finish instantly and wait. The barrier serializes the straggler.
- **No task overlap.** Even when task B only depends on 4 of task A's 16 tiles, all 16 must complete before any SM starts B.
- **Forces HBM round-trip between every task.** The barrier means the producer must commit results to global memory before any consumer can read them. No SMEM handoff possible.
- **Utilization = min(tiles, SMs) / SMs, per task, with no recovery.** Wasted cycles are gone — no other work fills the gap.

**Perf ceiling:** Bounded by `sum(max(task_compute, barrier_cost))` across all tasks. For small-tensor decode workloads (M=1), most tasks are barrier-dominated, not compute-dominated.

---

## Event-Driven (MPK-style)

**How it works:** Each SM pops tasks from a work queue (or is statically assigned). Tasks signal completion via atomics on per-dependency counters. When all producers for a dependency have signaled, the consumer task becomes runnable.

**Pros (perf):**
- **Zero barrier overhead.** No `grid.sync()`. A 0.5us elementwise op costs 0.5us + one atomic signal (~0.1us), not 0.5us + 5-10us barrier.
- **Task pipelining.** SM 0 finishes its tile of matmul A and immediately starts norm B while SM 15 is still computing its tile of A. Wall-clock = critical-path latency, not sum-of-max-per-stage.
- **Straggler tolerance.** Fast SMs move ahead instead of idling at the barrier. If 24 of 32 SMs finish attention early, they start the next task — the 8 stragglers catch up on their own timeline.
- **SMEM producer-consumer fusion becomes possible.** If you schedule a producer and its consumer on the same SM, the output can stay in shared memory — no HBM round-trip. This is the mechanism that enables real cross-task fusion.
- **Idle SMs do useful work.** When attention only needs 8 SMs, the other 24 can run independent tasks (the parallel Q/K/V matmuls, or even tasks from the next layer if dependencies allow).
- **Amortizes uneven tiling.** A task with 3 tiles on 32 SMs wastes 29 SMs under BSP. Under event-driven, those 29 SMs are already running the next task.

**Cons:**
- **Atomic contention.** Every task completion signals via atomics. Under heavy contention (many tasks signaling the same barrier), atomic throughput becomes a bottleneck. MPK mitigates this with GCD-based event granularity — coarsening signals so fewer atomics fire.
- **Work-queue overhead.** Popping tasks from a shared queue costs ~0.3-0.5us per pop (atomic CAS + potential contention). For very short tasks, this overhead is non-trivial relative to task compute.
- **Harder to profile.** Tasks overlap, SMs execute different tasks simultaneously — CUDA profiler traces become harder to interpret. Isolating one task's performance requires careful instrumentation.
- **Non-deterministic execution order.** Race conditions in the scheduler can cause subtle bugs. Debugging is harder — same inputs can produce different SM→task mappings across runs.
- **Register pressure from scheduler code.** The event-driven scheduler needs registers for task descriptors, dependency counters, and queue state. These registers are stolen from task code. With `__launch_bounds__(256, 1)`, you have max registers per thread — every register the scheduler uses is one the matmul can't.
- **L2 cache thrashing.** When SMs run different tasks simultaneously, they access different buffers, reducing L2 hit rates vs BSP where all SMs access the same buffers.
- **SMEM fusion is hard in practice.** The producer and consumer must share an SM, the SMEM layout must match, and the scheduler must be aware of the affinity. This is a significant implementation complexity for modest gains on small tensors.

**Perf ceiling:** Bounded by the critical path through the dependency graph. For a linear chain (most of transformer inference), this is `sum(task_compute)` with no barrier gaps. For parallel subgraphs (Q/K/V matmuls), it's `max(parallel_branch_latency)`.

---

## Head-to-Head on Performance

| Metric | BSP | Event-Driven | Winner |
|--------|-----|-------------|--------|
| **Per-task overhead** | ~5-10us (grid.sync) | ~0.1-0.5us (atomic signal + queue pop) | Event-driven, 10-50x less |
| **SM utilization on uneven tasks** | `min(tiles, SMs) / SMs` — no recovery | Idle SMs take other work | Event-driven |
| **Straggler impact** | Blocks all SMs until slowest finishes | Only blocks dependent tasks | Event-driven |
| **L2 cache behavior** | All SMs hit same buffers — high hit rate | SMs hit different buffers — lower hit rate | BSP |
| **Register budget for tasks** | Full budget, scheduler is trivial | Scheduler steals ~8-16 registers | BSP |
| **Memory traffic between tasks** | Always HBM round-trip | SMEM handoff possible (with effort) | Event-driven (theoretical) |
| **Throughput on wide task graphs** | Serialized, one task at a time | Parallel tasks run simultaneously | Event-driven |
| **Throughput on narrow chains** | ~Same compute, but with barrier tax | ~Same compute, no barrier tax | Event-driven (marginally) |

---

## The Transformer-Specific Reality

Transformers are mostly a linear chain — the task graph is maybe 2-3 tasks wide at best (Q/K/V matmuls, gate/up matmuls). So event-driven's advantage from parallelism is limited. The real wins for transformers specifically are:

1. **Barrier elimination** — 50-80 barriers × 5-10us = 250-800us saved. This is real, measurable, and scales with model depth.
2. **Straggler overlap** — the tail of one task overlaps the head of the next. For a 24-layer model with 10 tasks/layer, even 2us overlap per transition = 480us saved.
3. **Idle SM recovery on small tasks** — RMSNorm batch=1 (1 tile) runs on 1 SM. Under BSP, 31 SMs idle for ~3us + barrier. Under event-driven, 31 SMs are already doing the next matmul.

Rough estimate for SmolLM2-135M (current 8,856us megabake):
- Barrier elimination: saves ~300-500us
- Straggler overlap: saves ~100-200us
- Idle SM recovery: saves ~100-300us
- **Total: ~500-1000us → brings megabake to ~7,800-8,300us, roughly matching torch.compile**

That's the ~10-15% the design doc predicted. It gets you to parity on SmolLM2-135M but doesn't close the gemma-2b gap (that's attention, not scheduling).

---

## Verdict

Event-driven is strictly better for performance. BSP is strictly better for implementation simplicity. There is no workload where BSP is faster — it just has less code to get wrong.

For megabake specifically: event-driven scheduling alone gets you to roughly torch.compile parity on SmolLM2-135M, but fixing attention and tiling are still the bigger wins. The design doc's upgrade path is right — same `__device__` task functions, just swap the scheduler wrapper.
