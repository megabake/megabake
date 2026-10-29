"""Composable Blackwell GEMM using synchronous warp tensor-core MMA.

SM12x does not implement Hopper WGMMA. This provider admits SM10x and SM12x
without requiring the tcgen05/TMEM programming model. Every
warp keeps its fragments in registers; no asynchronous pipeline survives run().
"""
from __future__ import annotations

from dataclasses import replace
from math import prod

from ....graph import Region
from ....ir import Accumulator, Binary, Cast, Constant, Load, TensorProgram
from ....ops.gemm import GemmEpilogue
from ..generic import GeneratedContractionBody, _emit_expression, UnsupportedMapBody, generic_config
from ...interface import (
    BodyConfig, CandidateSet, ParticipantContract, ProviderRejection, ResourceEvidence,
    SelectionPolicy, TargetFacts, TileContract, validate_body_config,
)

BLACKWELL_ARCHITECTURES = frozenset({"sm100", "sm103", "sm120", "sm121"})


class BlackwellGemmProvider:
    provider_name = "blackwell_mma"

    def emit(
        self,
        program: TensorProgram,
        config: BodyConfig,
        *,
        defer_length_publish: bool = False,
        workers: int | None = None,
    ) -> GeneratedContractionBody:
        return emit_blackwell_gemm(program, config)

    def widen(self, program: TensorProgram, config: BodyConfig, threads: int) -> BodyConfig:
        if threads != config.participants.participating_threads:
            raise ValueError("the warp-MMA GEMM body runs four warps per CTA")
        return config

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        def reject(reason):
            return CandidateSet((), (ProviderRejection(self.provider_name, reason),))

        if target.architecture not in BLACKWELL_ARCHITECTURES:
            return reject("provider requires a Blackwell target")
        if not policy.allow_tuned:
            return reject("tuned providers disabled")
        if len(region.tensor_programs) != 1:
            return reject("provider requires a single tensor program")
        program = region.tensor_programs[0]
        fused = isinstance(region.semantics, GemmEpilogue)
        if fused and region.semantics.paired:
            return reject("paired GEMM epilogues require two accumulators")
        if (program.family != "contraction" or len(program.loads) < 2
                or program.reduction_axes != ("r0",)):
            return reject("provider requires a GEMM contraction")
        if fused and program.store.shape[-1] < 32:
            return reject("Blackwell GEMM epilogues require at least one full output tile")
        if any(domain.size < 1 for domain in program.iteration_domains):
            return reject("provider requires nonempty GEMM dimensions")
        dtype = program.loads[0].dtype
        if dtype not in {"torch.float16", "torch.bfloat16"} or program.loads[1].dtype != dtype:
            return reject("provider requires matching FP16 or BF16 operands and output")
        if program.store.dtype not in {dtype, "torch.float32"}:
            return reject("unsupported GEMM output dtype")
        left, right = program.loads[:2]
        plain_gemm = (
            len(program.store.shape) == 2
            and left.axes == ("o0", "r0")
            and right.axes in {("r0", "o1"), ("o1", "r0")}
        )
        broadcast_bmm = (
            len(program.store.shape) == 3
            and left.axes == ("o0", "o1", "r0")
            and right.axes == ("o0", "r0", "o2")
            and right.strides[0] == 0
        )
        if not (plain_gemm or broadcast_bmm):
            return reject("provider requires plain GEMM indexing")
        if any(stride < 0 for load in program.loads for stride in load.strides):
            return reject("negative operand strides are unsupported")
        # Reject scaled contractions and fused epilogues rather than silently
        # substituting plain GEMM for a different tensor program.
        acc = Accumulator("torch.float32")
        product = Binary("mul", Cast(Load(left.id, dtype), "torch.float32", "round_nearest_even"),
                         Cast(Load(right.id, dtype), "torch.float32", "round_nearest_even"), "torch.float32")
        if (program.accumulator_dtype != "torch.float32"
                or program.init != Constant(0.0, "torch.float32")
                or program.combine != Binary("add", acc, product, "torch.float32")
                or (not fused and (len(program.loads) != 2 or program.post != Cast(acc, dtype, "round_nearest_even")))):
            return reject("provider requires an unscaled FP32 accumulation and result cast")
        axis_names = (
            {"o0": "batch", "o1": "row", "o2": "n"}
            if broadcast_bmm
            else {"o0": "m", "o1": "n"}
        )
        try:
            _emit_expression(program.post, {load.id: index for index, load in enumerate(program.loads)},
                             axis_names, program.loads)
        except UnsupportedMapBody as error:
            return reject(str(error))
        base = generic_config(region, program)
        output_shape = region.semantics.output_shape if fused else program.store.shape
        return CandidateSet((replace(
            base,
            provider=self.provider_name,
            name="blackwell_mma_32x32x16_epilogue" if fused else "blackwell_mma_32x32x16",
            tiles=TileContract(
                cta_shape=(1, 32, 32) if broadcast_bmm else (32, 32),
                cluster_shape=(1, 1, 1) if broadcast_bmm else (1, 1),
                work_partition="persistent range of output tiles",
                mma_family="mma.sync", reduction_tile=(16,),
            ),
            participants=ParticipantContract(
                participating_threads=128, waiting_threads=128,
                warp_roles=("mma",) * 4, registers_per_thread=64,
            ),
            completion=replace(
                base.completion,
                register_fragments="warp MMA FP32 accumulators are cast and stored before return",
            ),
            evidence=ResourceEvidence(
                measured=False, source="megabake Blackwell warp MMA baseline",
                revision="blackwell-mma-v2", resources={
                    "threads": 128, "shared_bytes": 0,
                    "output_shape": output_shape, "broadcast_bmm": broadcast_bmm,
                },
                measurements={},
            ),
        ),), ())


def emit_blackwell_gemm(
    program: TensorProgram, config: BodyConfig
) -> GeneratedContractionBody:
    """Emit a synchronous 32x32 CTA tile with predicated M/N/K tails."""
    validate_body_config(config)
    if (config.provider != "blackwell_mma"
            or config.tiles.cta_shape not in {(32, 32), (1, 32, 32)}):
        raise ValueError("Blackwell GEMM requires the 32x32 warp MMA body contract")
    broadcast_bmm = bool(config.evidence.resources.get("broadcast_bmm"))
    rows, columns = prod(program.store.shape[:-1]), program.store.shape[-1]
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    dtype = "cutlass.BFloat16" if program.loads[0].dtype == "torch.bfloat16" else "cutlass.Float16"
    input_ids = tuple(dict.fromkeys(load.value_id for load in program.loads))
    indices = {load.id: input_ids.index(load.value_id) for load in program.loads}
    axis_names = ({"o0": "batch", "o1": "row", "o2": "n"} if broadcast_bmm
                  else {"o0": "m", "o1": "n"})
    post = _emit_expression(program.post, indices, axis_names, program.loads).code
    binding_type = ", ".join("cute.Tensor" for _ in range(len(input_ids) + 1))
    output_shape = tuple(config.evidence.resources.get("output_shape", program.store.shape))
    output_coordinates = ["n"]
    output_coordinate_lines = ""
    if len(output_shape) == 2:
        output_coordinates.insert(0, "m")
    else:
        output_coordinate_lines = "                output_row = m\n"
        for index, size in reversed(tuple(enumerate(output_shape[:-1]))):
            coordinate = f"output_o{index}"
            output_coordinate_lines += f"                {coordinate} = output_row % {size}\n"
            output_coordinate_lines += f"                output_row = output_row // {size}\n"
            output_coordinates.insert(0, coordinate)
    output_index = ", ".join(output_coordinates)
    if broadcast_bmm:
        batch_rows = program.store.shape[1]
        tile_coordinate_line = ""
        coordinate_lines = (f"                    batch = m // {batch_rows}\n"
                            f"                    row = m % {batch_rows}\n")
        left_index, right_index = "batch, row, k", "0, k, n"
    else:
        tile_coordinate_line = ""
        coordinate_lines = ""
        left_index = "m, k"
        right_index = "(k, n)" if program.loads[1].axes == ("r0", "o1") else "(n, k)"
    ntiles = (columns + 31) // 32
    source = f'''import cutlass
import cutlass.cute as cute

@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, 16)), (2, 2, 1))
    thr_mma = mma.get_slice(thread)
    ca = thr_mma.partition_A(cute.make_identity_tensor((32, 16)))
    cb = thr_mma.partition_B(cute.make_identity_tensor((32, 16)))
    cc = thr_mma.partition_C(cute.make_identity_tensor((32, 32)))
    fa = cute.make_rmem_tensor(ca.shape, {dtype})
    fb = cute.make_rmem_tensor(cb.shape, {dtype})
    fc = cute.make_rmem_tensor(cc.shape, cutlass.Float32)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        tile_id = cutlass.Int32(work_iterator[tile])
        row_base = (tile_id // {ntiles}) * 32
        col_base = (tile_id % {ntiles}) * 32
{tile_coordinate_line}        fc.fill(0.0)
        for kbase in cutlass.range(0, {reduction}, 16):
            for i in cutlass.range_constexpr(cute.size(fa)):
                m = row_base + ca[i][0]
                k = kbase + ca[i][1]
                fa[i] = {dtype}(0.0)
                if m < {rows} and k < {reduction}:
{coordinate_lines}                    fa[i] = tensor_bindings[0][{left_index}]
            for i in cutlass.range_constexpr(cute.size(fb)):
                n = col_base + cb[i][0]
                k = kbase + cb[i][1]
                fb[i] = {dtype}(0.0)
                if n < {columns} and k < {reduction}:
                    fb[i] = tensor_bindings[{indices[program.loads[1].id]}][{right_index}]
            cute.gemm(mma, fc, fa, fb, fc)
        for i in cutlass.range_constexpr(cute.size(fc)):
            m = row_base + cc[i][0]
            n = col_base + cc[i][1]
            if m < {rows} and n < {columns}:
{output_coordinate_lines}                accumulator = fc[i]
                tensor_bindings[{len(input_ids)}][{output_index}] = {post}
'''
    return GeneratedContractionBody(
        source=source, threads_per_cta=128, output_tile_elements=512,
        reduction_chunk=16, tile_count=((rows + 31) // 32) * ntiles, work_step=1,
        work_iterator_encoding="one row-major 32x32 output tile ordinal per CTA",
        phases=("output_tile", "warp_mma", "epilogue"),
    )
