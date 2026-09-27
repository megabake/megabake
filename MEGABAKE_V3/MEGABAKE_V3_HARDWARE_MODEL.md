# MegaBake V3: target facts and SM-qualified compilation

Status: proposed backend contract, revised 2026-09-27. No V3 target query or GPU calibration is implied. The [architecture](MEGABAKE_V3_ARCHITECTURE.md) owns result scope; this file owns target identity, legality and cost provenance.

## 1. A profile for decisions, not a hardware-discovery project

A DFS can traverse a device description that already exists; it cannot discover undocumented bandwidth, contention or all memory latencies. Use a compact backend-qualified `TargetProfile` containing only facts that change body legality, movement, synchronization, residency, layout, placement or measured cost. The common indexed program does not branch on CUDA SM numbers or space-name strings.

Separate three classes of facts:

| Class | Source | Use |
|---|---|---|
| Queried capability | Runtime/driver/compiler APIs for the actual device and visible partition | Hard legality and resource bounds |
| Documented feature | Versioned CUDA/PTX/CUTLASS/cuBLASDx contracts | Instruction, descriptor, collective and compatibility legality |
| Calibrated cost | Exact-shape/stage/whole-entry measurements with provenance | Candidate ranking and tuning, never a correctness proof |

An unknown cost stays `UNKNOWN`. A product-sheet peak is a bound or context, not achieved bandwidth. A body compiled on one SM or CUDA version does not carry a performance measurement to another.

## 2. Common schema and backend boundary

```text
TargetProfile {
  backend_and_target_identity, visible_partition_or_mesh,
  runtime_driver_compiler_and_provider_versions,
  execution_domains_and_capacity, feature_sets,
  address_spaces_and_reachability, movement_edges_and_issuers,
  synchronization_scopes_visibility_and_progress,
  collective_topology, descriptor_and_stage_requirements,
  resource_limits_and_query_provenance,
  calibrated_costs: [key, value, unit, conditions, uncertainty, source],
  unknown_fields
}
```

Spaces have backend-qualified stable IDs such as `cuda:cta_shared`; the common compiler compares declared properties rather than spellings. A logical requirement can request local forwarding, an async copy with distinct completion and source-retirement tokens, or a collective group. The adapter binds it to a real mechanism or rejects that candidate. It also **proposes** target-native tile refinements, body tactics and schedules; it is not merely a feature table queried after common planning.

A future TPU adapter would describe TensorCore/mesh domains, VMEM/SMEM/HBM, DMA/semaphore/remote-copy protocols and topology-specific collectives as its own mechanisms. It would not rename VMEM to CUDA shared memory or inherit CUDA cohort limits. Inferact's [gridless Pallas program](https://inferact.ai/blog/tpu-megakernels) motivates this boundary but is not a MegaBake backend.

## 3. CUDA target resources

| Resource/space | Relevant property | Important limit |
|---|---|---|
| Registers | Thread/warp accumulators and live values | Entry-wide allocation and spills affect residency |
| CTA shared memory | Cooperative tiles and staged loads | Reachable only by the CTA without special protocols |
| Cluster distributed shared memory | Optional cluster exchange | Requires supported launch, participation and lifetime |
| Global address space | Weights, state and inter-CTA values | Physical traffic may hit L2 rather than HBM |
| L2 | Hardware-managed shared cache | Residency/contending traffic depend on workload |
| Tensor memory where available | Specialized MMA accumulator storage | Architecture-specific allocation/access contract |
| Instruction footprint | Body mixture and template unrolling | Code-cache pressure and compilation time |

At device/session initialization query compute capability, visible SM count, block/thread limits, register and shared-memory limits, opt-in dynamic shared memory, cooperative/cluster support and any selected feature attributes. CUDA's [device property API](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html) is the starting point. A MIG partition changes visible resources; it is not a product-name constant.

After compiling the **complete selected entry**, read function attributes and compiler resource reports; set required launch attributes; ask CUDA occupancy/cooperative APIs for the actual block size and dynamic shared memory. For an ordinary cooperative grid, admitted worker CTAs are bounded by visible SMs times supported active CTAs per SM for that exact entry. Allocation granularity, launch bounds and target restrictions require the API result rather than hand division. A worker is not pinned to a physical SM. [CUDA cooperative groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)

A shared-memory scratch union can reuse bytes across exclusive task lifetimes, but the entry's requested peak storage still constrains occupancy. Register allocation and code footprint are also whole-entry properties. Long-lived streamed-reduction accumulators, extra cuBLASDx roles and next-task preloads can change the resource envelope. Measure the actual mixture rather than `max(isolated_body_cost)`.

## 4. Feature sets across SM generations

CUDA's baseline features, architecture-specific `_a` features and family-specific `_f` features have different compatibility guarantees. `_a` code runs only on its exact compute capability; `_f` code runs within its documented family. Numeric `SM >= 90` is not a safe proxy for TMA/WGMMA/tcgen05 support. [CUDA compute-capability guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

| Candidate family | Expected implementation distinction | Planning consequence |
|---|---|---|
| Ampere/Ada | Warp MMA and target-supported async-copy variants | Target-specific tile/mainloop and resources |
| Hopper `sm_90a` | TMA, WGMMA and architecture-specific barriers | Dedicated body and descriptor/participation proof |
| Data-center Blackwell `sm_100a` | tcgen05/TMEM mechanisms | Separate body and accumulator/storage protocol |
| Other Blackwell targets such as `sm_120` | Different available feature set | Never assume an `sm_100a` body works unchanged |
| Later targets | Documented baseline/family/architecture features | Add provider variants and local tuning evidence |

This table names tactic families, not a closed list of supported GPUs. CUTLASS and cuBLASDx versions may support additional feature sets, but an available header is not proof that one generated body is fast. Maintain a portable SIMT correctness path for the declared supported subset, then measure accelerated providers per target. cuBLASDx's current [requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html) include CUDA 13.0+, so the provider registry must handle older toolkit environments without importing an incompatible header.

## 5. Body and plan candidate keys

A body alternative is keyed by indexed semantics, numerical policy, shape/stride/alignment guard, algorithm, tile/layout, feature set, compiler/provider version and its physical participant/stage contract. The physical plan adds visible resource profile, body mixture, storage/liveness, CTA count/block size, scheduler policy, descriptor bindings and code-generation version.

A correctness artifact key and a performance-tuning key differ. Two devices with a documented compatible cubin may share correctness code while requiring different worker counts or measured tactic ranking. A new MIG profile can preserve instruction legality yet change occupancy and bandwidth. A CUDA/toolchain update may alter register allocation, assembly and provider behavior; invalidate affected measurements and admission reports. No disk cache entry may silently relabel an old result as current target evidence.

## 6. Historical H200 MIG facts and their limit

The latest V2 report described an H200 `3g.71gb` partition with 60 visible SMs; an older tracked record described `2g.35gb` with 32. Keep these as distinct historical contexts. NVIDIA's [MIG profile table](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html) gives resource fractions, not achieved bandwidth. The full H200 [product specification](https://www.nvidia.com/en-us/data-center/h200/) lists 4.8 TB/s; proportional partition arithmetic is only an estimate. Neither number sets V3 CTA count, shared-memory allocation or body preference.

The existing V2 launcher reserved a large fixed shared-memory footprint and fixed worker count, and its compiled entry had severe occupancy/stack costs in the [GPU audit](MEGABAKE_V3_GPU_REANALYSIS.md). V3 must derive each launch from selected target and compiled entry. A 128- or 256-thread example is only a candidate for one body mixture, not a compiler invariant.

## 7. Minimum calibration campaign

Calibrate only what the current workload/candidate set uses:

1. Strongest equivalent `torch.compile`/cuBLAS/cuBLASLt path for each hot exact shape, dtype, layout and cache condition.
2. Generated SIMT and tensor-core tactics standalone and in a lean mixed persistent entry, with padding/packing/setup and compiled resources.
3. Whole-entry empty/minimal overhead, grid joins, publication/acquire, descriptor binding and output ownership.
4. Representative attention/state costs at short and long valid context.
5. Activation handoff, L2/global traffic and selected cross-task/preload pairs under measured contention.
6. Tile-tail sweeps around the actual resident-worker count and selected static/dynamic schedule costs.

The planner may use priors to explore, but a winner requires measured full invocation latency. No GPU means source/plan legality may be analyzed and cross-compilation may reveal syntax/resource issues; runtime correctness, residency and speed remain unestablished. [Performance protocol](MEGABAKE_V3_PERFORMANCE_MODEL.md)

## 8. Target admission procedure for an implementation agent

For one selected source/plan candidate, perform the following in order:

1. Resolve target identity from the **actual selected device/visible partition**: compute capability, visible SMs, cooperative support, block/thread/shared/register limits, relevant feature attributes, runtime and driver versions. Store provenance for each queried or documented fact.
2. Check each body provider's exact feature/toolchain/precision/layout guards. The provider must name the instruction family it needs. A numeric `SM >= 90` condition is insufficient; `_a` and `_f` targets follow the documented compatibility set. [CUDA feature sets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
3. Choose one enclosing block/cluster shape accepted by **every** selected body and one binding ABI for all descriptors. Bound kernel argument bytes, static/dynamic shared memory and required launch attributes. A cuBLASDx pipeline may require its `get_block_dim()`; an incompatible mixture is a plan rejection, not a request to launch an extra grid.
4. Emit and compile the **whole selected entry**. Read compiled function attributes and compiler resource/spill report. Recompute shared allocation, active blocks per SM and maximum cooperative worker grid using the appropriate CUDA APIs for this function and dynamic shared-memory setting. Never substitute the sum or maximum of isolated body resource estimates.
5. If source compilation, feature legality, resource limits, collective participation or residency fails, record a diagnostic and request another tactic/tile/plan. Only a successful admitted entry may be launched. Device correctness and performance still need separate evidence.

`TargetExecutionPlan` therefore has lifecycle states `lowered`, `compiled`, `admitted`, `measured`; a compilation result can invalidate the plan's earlier resource estimate. A source hash without an actual cubin/function attribute report is not `admitted`. A CPU mock profile can check guard logic, but cannot establish real residency.

## 9. Current checkout versus proposed V3 backend

The legacy `src/megabake/runtime/cuda_compiler.py` concatenates all task `.cu` sources, compiles with `--use_fast_math`, and selects `sm_${sm}a` for every numeric SM at least 90. Those decisions can bring unrelated code into the resource envelope, change the declared numerical policy or form an invalid target name. The V3 source builder emits only selected bodies, uses policy-qualified math flags and selects an exact documented feature target. Preserve the legacy compiler as historical control while bringing up `src/megabake/v3/backends/cuda/` or an equivalent isolated backend.

The package's pinned `requirements.txt` environment uses a CUDA 12.4 PyTorch wheel and CUTLASS 3.8.0.0. A wheel version is **not** proof of the selected nvcc or driver; query them. Current cuBLASDx documents require CUDA Toolkit 13.0+ and CUTLASS 4.4.1+, so its provider cannot be assumed to exist in the pinned lane. [NVIDIA cuBLASDx requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html) A body from a CUDA13 experiment carries that version in its candidate and artifact key; it cannot be silently reused by the CUDA12 main build.

## 10. Target-profile example and failure behavior

A profile record may state `backend=cuda`, `compute_capability=90`, `feature_target=sm_90a`, `visible_sms=60`, `cooperative=true`, `nvcc=12.8`, `runtime/driver=queried`, and explicitly unknown calibrated bandwidth. Those numbers are an **illustrative historical H200 MIG context**, not a target auto-detection rule. A second device with 132 SMs or a different MIG partition changes worker count and tuning even if its instruction legality matches. Report `unknown` rather than insert product-sheet bandwidth into a measured-cost field.

| Failure | Correct compiler behavior |
|---|---|
| Selected body needs unavailable WGMMA/TMA or tcgen05 | Reject body, try legal SIMT/MMA variant |
| Compile returns spills or shared use beyond budget | Re-rank/recompile smaller tactic or materialized schedule |
| cuBLASDx handle needs incompatible block shape | Reject mixed strict plan or use a different body; external plan stays separate |
| Cooperative occupancy admits fewer workers than planned | Replan grid/work assignment and reverify progress before launch |
| Target/toolkit changed since cache creation | Invalidate resource/tuning evidence; rebuild or verify compatible correctness artifact |
