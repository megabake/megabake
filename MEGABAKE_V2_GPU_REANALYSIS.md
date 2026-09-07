# MegaBake V2.3 GPU Reanalysis

Status: first empirical backend case study, 2026-09-07. This document records H200 results used to
test and revise the generic V2 compiler contracts. H200 is the available laboratory machine, not
the intended primary deployment target. This document is normative with:

- [`MEGABAKE_V2_ARCHITECTURE.md`](./MEGABAKE_V2_ARCHITECTURE.md)
- [`MEGABAKE_V2_DATAFLOW_DIAGRAM.md`](./MEGABAKE_V2_DATAFLOW_DIAGRAM.md)
- [`MEGABAKE_V2_IR_AND_REUSE_PLAN.md`](./MEGABAKE_V2_IR_AND_REUSE_PLAN.md)
- [`grace_hack.md`](./grace_hack.md)

The north star remains strict and measurable:

> For a supported target/model/bucket, emit one owned GPU compute grid whose complete invocation
> beats the strongest equivalent `torch.compile` mode. Report strict-one-grid coverage separately
> from hybrid fallback coverage across the model and hardware matrix.

This means a portfolio of generated one-grid artifacts. It does not mean one universal binary,
one launch envelope, or one linear algorithm for every model and GPU.

The evidence has two different scopes:

| Portable compiler conclusion | H200-only observation |
|---|---|
| logical work and resident workers require separate contracts | the measured grid has 60 resident CTAs |
| compiled entry resources must gate composition | this entry uses 184 registers/thread and 225.28 KB SMEM |
| exact shape and target choose the linear family | the listed `gemvx`/`gemv2T`/WMMA crossovers |
| strongest equivalent baseline gates acceptance | the measured 1.782-ms compiled baseline |
| semantic and physical bandwidth are distinct | the inferred 2.4-TB/s MIG share and measured percentages |

Only the left column enters target-neutral compiler policy. The right column is stored under the
H200 `TargetFingerprint`; another backend may use it to order experiments but must regenerate,
resource-check, and retune on its own hardware.

## 1. Environment and evidence boundary

The measurements in this addendum were taken on:

```text
GPU              NVIDIA H200 MIG 3g.71gb
compute           SM90, 60 visible SMs, approximately 29.4 MiB visible L2
driver            595.58.03
PyTorch           2.6.0+cu124
PyTorch CUDA      12.4
nvcc              12.8.93
Nsight Compute    2025.1.1
Transformers      5.16.1
model             HuggingFaceTB/SmolLM2-135M
workload          reference FP16-like decode, batch=1, sequence=1
```

Clocks could not be locked under MIG. Nsight Compute was run with `--clock-control none`; the full
MegaBake capture reported a 1.98 GHz SM clock. Durations are therefore evidence for this instance,
not portable constants. Nsight replay can change cache state, so its resource and instruction
measurements are stronger evidence than replayed microkernel latency.

No correctness repair was attempted. The reported maximum logit differences (`0.195312` for
MegaBake and `0.484375` for the ordinary compiled path) are not by themselves an adequate
correctness or model-quality proof. Performance conclusions below are conditional on the reference
policy eventually passing its numerical gate.

## 2. What the first end-to-end result really says

The original run reported:

| Backend | End-to-end | Counted GPU kernels | Counted GPU kernel time |
|---|---:|---:|---:|
| eager | 11,350.7 us | 674 | 1,491.0 us |
| ordinary `torch.compile` | 5,691.4 us | 424 | 985.1 us |
| MegaBake | 4,845.5 us | 1 | 4,748.7 us |

The correct conclusions are:

1. MegaBake successfully removed CPU launch serialization relative to the ordinary compiled path.
2. Its single compute grid is about 4.8 times slower than the sum of the compiled baseline's GPU
   kernels (`4,749 / 985`).
3. The 1.17x end-to-end win is not the north-star comparison because ordinary `torch.compile` was
   still paying roughly 4.7 ms of host-side launch delay.
4. Through the same export route, `torch.compile(mode="reduce-overhead")` measured 1,781.7 us median
   and 1,737.9 us best, with 1,026.2 us summed CUDA-node duration. MegaBake is currently about 2.7x
   slower than this measured low-overhead baseline.

The new immediate target is therefore not “preserve the 1.17x lead.” It is to remove roughly
3 ms or more from the one-grid device program while retaining its launch advantage.

## 3. Measurement corrections

### 3.1 The printed MIG bandwidth and roofline are wrong

The harness hard-codes 500 GB/s for any H200 MIG. NVIDIA specifies 4.8 TB/s for H200 and specifies
that `3g.71gb` owns 4/8 of the memory and L2 slices. The proportional peak is therefore about
2.4 TB/s, an inference from those two public specifications, not 500 GB/s.

For 269 MB of semantic parameter bytes:

```text
product-peak floor at 500 GB/s       538 us    # old report
partition-proportional floor         112 us    # 269 MB / 2.4 TB/s
MegaBake semantic effective BW        56 GB/s  # 269 MB / 4.77 ms
MegaBake fraction of 2.4 TB/s        2.35%
Nsight Compute DRAM throughput       2.33%
```

The agreement between the last two values is strong evidence that the proportional interpretation
is appropriate for this run. The 112 us value remains only a mathematical product-peak floor; it is
not a prediction for hundreds of differently shaped phases.

V2.3 records three different quantities and never calls them all “bandwidth utilization”:

```text
semantic_effective_bw_e2e = semantic bytes / complete invocation time
semantic_effective_bw_body = semantic bytes / relevant GPU active interval
physical_dram_bw = hardware-counter bytes or throughput over the measured interval
```

Semantic bytes and physical DRAM bytes can differ because of cache hits, redundant reads, spills,
workspace traffic, and replay state. A useful calibrated target is the sum of cold, model-context
shape-family baselines—not product peak alone.

### 3.2 “One kernel” hid five CUDA copy operations

An unfiltered warm MegaBake invocation contained:

```text
Memcpy DtoD   0.833 us
Memcpy HtoD   0.737 us
Memcpy DtoD   1.408 us
Memcpy DtoD   1.504 us
megakernel  4771.715 us
Memcpy DtoD   1.024 us
```

The copies are small today, but filtering them out made the operation count and the residual
`Host(us)` column ambiguous. V2.3 reports at least:

```text
host submissions
compute grids
all GPU operations
memcpy/memset operations and bytes
complete GPU interval
CPU enqueue and framework time
```

The strict north star permits explicit binding/reset/output copies only if its contract permits
them, but it never reports them as nonexistent. `RUN_INTO`, borrowed output, preallocated state,
and static bindings remain the preferred steady-state contract.

### 3.3 The strongest baseline is mandatory

The measured baseline ladder for this exact bucket is now:

```text
ordinary torch.compile                         5691 us
export + torch.compile(mode="reduce-overhead") 1782 us
MegaBake current                               4846 us
```

The precise baseline mode, capture status, graph breaks, addresses, output ownership, and operation
timeline belong in `MeasuredBaselineSuite`. A win over the first row is not accepted when the
second row is legal.

## 4. Root cause in the current cubin

Nsight Compute reported for the complete MegaBake grid:

| Property | Measured value |
|---|---:|
| grid and block | 60 CTAs x 256 threads |
| registers/thread | 184 |
| dynamic shared memory/CTA | 225.28 KB |
| stack frame/thread | 1,072 bytes |
| register-limited CTAs/SM | 1 |
| shared-memory-limited CTAs/SM | 1 |
| theoretical/achieved occupancy | 12.5% / 12.5% |
| compute throughput | 6.42% |
| DRAM throughput | 2.33% |
| duration | 4.77 ms |

`cuobjdump` independently reports `REG:184` and `STACK:1072`. The SASS contains many `LDL`/`STL`
instructions. This does not prove that every local instruction executes on the decode path, but it
does prove that “zero reported spill bytes” is not sufficient evidence of a stack-free hot path.
The runtime-sized `float acc[64]` in the skinny implementation is exactly the kind of construct that
must disappear from generated `M=1`, `M=2`, and `M=4` entries.

The universal translation unit makes prefill WGMMA, attention, reductions, interpreted pointwise,
and skinny decode reachable from one entry. Their maximum register, stack, block-shape, and shared-
memory requirements become a tax on the whole grid. V2.3 makes reachability pruning and entry-
envelope compilation an acceptance gate, not a later cleanup.

## 5. The current skinny mapping is the wrong abstraction

For `M<=4`, the current code:

- launches at most one resident CTA per visible SM;
- sets the logical tile count equal to the resident worker count;
- divides N into one range per worker;
- assigns one output column to one thread;
- makes that thread perform the whole K reduction serially;
- reserves almost all per-CTA shared memory for double-buffered weights.

For common SmolLM shapes this leaves only about 10 valid output threads per CTA at `N=576`, and
about 26 at `N=1536`, while every valid thread serially reduces K. A CTA-count sweep showed that
small shapes saturated before 60 CTAs; more resident workers did not repair the reduction mapping.

Normal-profiler isolated measurements, used only as mapping evidence because the small operands can
be L2-hot, were:

| `(M,N,K)` | vendor-selected kernel | vendor | current MegaBake, 60 tiles |
|---|---|---:|---:|
| `(1,576,576)` | cuBLAS `gemvx` | 2.63 us | 15.49 us |
| `(1,1536,576)` | cuBLAS `gemv2T` | 3.17 us | 15.97 us |
| `(1,576,1536)` | cuBLAS `gemvx` | 2.94 us | 33.22 us |
| `(1,49152,576)` | CUTLASS WMMA | 34.98 us | 77.76 us |
| `(1,4096,4096)` | tensor-core GEMM | 23.39 us | 95.78 us |

Nsight resource profiles reveal three different vendor mappings:

| Shape/family | Grid | Block | Reg/thread | Dynamic/static SMEM | Theoretical occupancy |
|---|---:|---:|---:|---:|---:|
| `1x576x576` `gemvx` | 144 | 128 | 162 | 528 B / 0 | 18.75% |
| `1x1536x576` `gemv2T` | 192 | 128 | 58 | 0 / 2.56 KB | 50% |
| `1x49152x576` WMMA | 3,072 | 32 | 72 | 4.61 KB / 0 | 43.75% |

The vocabulary head deliberately uses tensor cores even though `M=1`. The ordinary projections use
K-parallel warp/CTA reductions with many more logical CTAs than visible SMs. Therefore `M<=4` is
not a kernel-family decision, and `num_tiles = num_sms` is not a valid work model.

V2.3 separates:

```text
resident_worker_count       # cooperative CTAs kept resident
logical_work_count          # output/reduction tiles for the phase
logical_to_worker_mapping   # static striding, ranges, or measured cursor
entry_launch_envelope       # threads, warps, SMEM, register/stack ceiling
```

A persistent worker can process several logical tiles. If the generated entry uses sufficiently
few resources, the cooperative grid may also place multiple resident CTAs on each SM. Neither count
is automatically the visible SM count.

## 6. Revised decode-linear portfolio

The first portfolio should contain at least:

1. **Warp-reduction GEMV.** Lanes traverse K cooperatively, reduce partial sums, and compute one or
   several output rows per warp. Candidate blocks contain 4–8 warps, with logical work striding over
   N. This is the first candidate for small and medium projection widths.
2. **CTA/split-K reduction GEMV.** Use multiple warps per output or output group when K is large
   enough that one-warp latency dominates. Store no global partial unless a measured split-K win
   pays for its reduction.
3. **Padded-M tensor-core projection.** Compute a tensor-core tile with unused M rows when large N
   makes bandwidth/coalescing gains exceed wasted math. The 49,152-row vocabulary head proves this
   crossover exists.
4. **Mixed-dtype streaming variants.** W8A16/W4A16 and FP8 are separate numerical contracts, but
   they are eventually the mechanisms that lower the semantic weight-byte floor.

Each applicable tactic is sourced through the same four-family tournament: cuBLASDx descriptors,
CUTLASS collectives, direct CuTe compositions, and MegaBake-native CUDA. cuBLASDx candidate coverage
is mandatory for every target/shape/precision cell supported by the pinned release.

For strict one-grid decode, these families must be generated into a compatible entry envelope. A
128-thread common envelope with phase-specific warp participation is a concrete starting hypothesis,
not a mandated answer. The prefill WGMMA entry is not reachable from this decode artifact.

Candidate selection keys include exact M/N/K, layout, alignment, epilogue, storage precision, MIG
resource fractions, and complete entry resources. The compiler learns the crossover; it does not
encode “all M=1 uses GEMV.”

## 7. Scheduling priority after measurement

Switching the existing model between the per-SM dependency queues and the simple static grid loop
gave:

```text
per-SM queues       4.756 ms kernel
static grid loop    4.573 ms kernel
```

The generic scheduler costs about 4% here. Replacing it with `PhaseProgram` remains correct, but it
cannot explain or close the 2.7x gap to the low-overhead compiled baseline. Work order is therefore:

1. specialize the entry and make the hot linear bodies competitive;
2. remove stack/local behavior and universal resource reachability;
3. form transformer composites that remove materializations and joins;
4. simplify the phase executor and invocation housekeeping;
5. only then evaluate more elaborate overlap or serving schedulers.

The saved 463-task diagnostic also shows the non-linear tail cannot be ignored: copies, RoPE,
reductions, attention, and pointwise work collectively account for a large part of recorded task
cycles. Matching vendor GEMV alone is necessary but not sufficient; semantic composites must make
many of those task boundaries disappear.

## 8. Prelude, preload, and cuBLAS extraction verdict

### 8.1 GraCE prelude

A GraCE prelude updates bytes in the launch-parameter storage of a captured vendor graph node. It
does not extract code, turn a kernel entry into a device function, or remove the target grid. The
paper itself places the prelude and vendor kernel in separate graph nodes and reports that prelude
indirection can exceed 10 us as indirect argument count grows.

It is useful for a dynamic-binding hybrid graph baseline. It cannot satisfy strict one-grid
execution.

### 8.2 `LD_PRELOAD`, CUPTI, and a live `CUfunction`

If “preload” means interposition, it can observe the exact host launch, live `CUfunction`, geometry,
attributes, and raw argument values. That enables:

- a vendor oracle keyed by exact shape and environment;
- a frozen direct-host-relaunch experiment that bypasses repeated cuBLAS dispatch;
- comparison of owned SASS/resources with the selected private entry.

The handle is still a host-side opaque kernel handle. Relaunch still creates a separate grid. A
fully linked private cubin is not relocatable device-function code, and neither
`cuLibraryGetUnifiedFunction` nor dynamic parallelism converts it automatically.

### 8.3 Device-launched graph or dynamic parallelism

These can move control to the GPU, but the vendor work remains a child/separate grid. A device graph
also has fixed structure and documented update/upload restrictions. It may be a useful multi-grid
decode controller, but its result must be reported as multiple GPU grids.

### 8.4 cuBLASDx

cuBLASDx is the supported NVIDIA answer for selected BLAS operations inside a user kernel. It is a
separate header-only product and does not expose private cuBLAS kernels; NVIDIA's cuBLAS manual says
it is not shipped with the CUDA Toolkit. It is not installed in this environment, so no local
cuBLASDx performance claim is made.

Current GEMM execution is block-level, which is compatible with—but does not replace—MegaBake's
persistent outer scheduler. A resident CTA obtains a logical output tile, uses a cuBLASDx descriptor
for cooperative tile execution, fuses an epilogue through the register accumulator when profitable,
and then advances to another logical tile. MegaBake still owns global tiling, K traversal/pipeline,
phase dependencies, and entry resources.

Provision it as a pinned first-class dependency, then broadly enumerate register-fragment,
shared-memory, and pipelined device-GEMM descriptors. Measure them standalone, inside a minimal
persistent worker, inside composites, and in separately compiled whole-entry beams. Its own guidance
emphasizes pipelining, suggested layouts, 128-bit alignment, enough blocks, and merging adjacent
memory-bound work. It competes with CUTLASS collectives, direct CuTe compositions, and native CUDA
for each decode shape; search coverage is mandatory where supported, but the winner is measured.

### 8.5 Production decision

Do not spend the critical path extracting private cuBLAS cubins. Spend a bounded research effort on
launch tracing because it is a valuable oracle. Production `DEVICE_CALLABLE` bodies are selected
by a first-class tournament across cuBLASDx, CUTLASS collectives, direct CuTe, and MegaBake-native
CUDA.

## 9. Falsifiable V2.3 gates

### Gate A — measurement truth

- detect MIG memory/SM/L2 fractions and calibrate physical bandwidth;
- report unfiltered GPU operations and bytes;
- store ordinary and strongest low-overhead compiled baselines;
- distinguish end-to-end semantic, GPU-body semantic, and counter-derived physical bandwidth.

### Gate B — entry-envelope proof

- generate a decode-only cubin with no prefill/attention-heavy unreachable path;
- record register count, stack frame, local transactions, SMEM, and resident CTAs;
- reject unexpected `LDL/STL` on the hot skinny path;
- demonstrate more than one CTA/SM when that variant's measured mapping needs it.

### Gate C — hot-shape parity

- classify all model linears by exact signature and frequency;
- compare cold/model-context latency and physical bytes with the vendor-selected kernel;
- reach at least 80% of vendor body performance before whole-model composition;
- reach 95%, or prove the remaining deficit is repaid by a measured fused composite.

### Gate D — strict model win

- exactly one owned compute grid for the supported bucket;
- complete invocation beats the strongest equivalent compiled mode by a noise margin at p50 and
  does not regress p99;
- correctness and output-lifetime contracts pass;
- every remaining copy, reset, join, and idle interval is attributed.

### Gate E — reliability claim

Publish a matrix, not an anecdote:

```text
models x GPU/driver/toolkit x decode/prefill bucket x numerical policy
  -> strict-one-grid legal?
  -> strict-one-grid wins?
  -> speedup distribution
  -> hybrid fallback result
```

“Reliably on most models and most hardware” means a predeclared coverage threshold and no hidden
fallbacks in the strict score. Hybrid fallback is still important product behavior, but it is a
different metric.

An H200-only matrix cannot satisfy this gate. The first portability cell should be a non-Hopper
CUDA family such as SM80/A100, using the same target-neutral regions but its own legal candidates,
calibration, compiled resources, packed layouts, and measurements. Broader architecture families
are added explicitly rather than inferred from either result.

## 10. Primary sources for the revision

- [GraCE OSDI'26 paper](https://www.usenix.org/system/files/osdi26-ghosh.pdf)
- [CUDA Graph device launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-launch)
- [CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
- [CUDA Driver library management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html)
- [cuBLAS and cuBLASDx distinction](https://docs.nvidia.com/cuda/cublas/)
- [cuBLASDx overview](https://docs.nvidia.com/cuda/cublasdx/)
- [cuBLASDx performance guidance](https://docs.nvidia.com/cuda/cublasdx/performance.html)
- [CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
- [H200 product bandwidth](https://www.nvidia.com/en-us/data-center/h200/)
- [H200 MIG profile fractions](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html#h200-mig-profiles)

## 11. Bottom line

This H200 case study validates the one-grid motivation and invalidates the current universal-entry
strategy on that target. Launch fusion removed several milliseconds of host delay, but a 60-CTA,
256-thread, 225-KB, 184-register entry turned the GPU into the bottleneck. Prelude kernels cannot
import cuBLAS SASS into that entry. The generic lesson is to retain target-neutral regions and
contracts, then let each target backend generate feature-legal exact-shape bodies, compile a
compatible envelope, calibrate and autotune on the actual device, and accept it only against that
device's low-overhead baseline. The H200 tactics and numbers are evidence for one scorecard cell,
not global defaults.
