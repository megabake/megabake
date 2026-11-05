"""SM12x synchronous warp-MMA GEMM provider."""

from dataclasses import replace
from typing import Any

from ....graph import Region
from ...interface import (
    BodyConfig,
    CandidateSet,
    ProviderRejection,
    ResourceEvidence,
    SelectionPolicy,
    TargetFacts,
)
from ..sm100.gemm import BlackwellGemmProvider, emit_blackwell_gemm

SM12X_ARCHITECTURES = frozenset({"sm120", "sm121"})


class SM12xGemmProvider(BlackwellGemmProvider):
    provider_name = "sm12x_mma"

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        if target.architecture not in SM12X_ARCHITECTURES:
            return CandidateSet(
                (), (ProviderRejection(self.provider_name, "provider requires an SM12x target"),)
            )
        compatible_target = replace(target, architecture="sm100")
        candidates = super().get_candidates(region, compatible_target, policy)
        mma = tuple(
            replace(
                config,
                provider=self.provider_name,
                name="sm12x_mma_16x32x16",
                evidence=ResourceEvidence(
                    measured=False,
                    source="megabake SM12x warp MMA baseline",
                    revision="sm12x-mma-v1",
                    resources=config.evidence.resources,
                    measurements={},
                ),
            )
            for config in candidates.configurations
        )
        gemv = ()
        if mma and region.tensor_programs[0].store.shape[0] == 1:
            config = mma[0]
            gemv = (
                replace(
                    config,
                    name="sm12x_gemv_warp4",
                    tiles=replace(
                        config.tiles,
                        cta_shape=(1, 4),
                        work_partition="four output columns per CTA, one per warp",
                        mma_family=None,
                        reduction_tile=(32,),
                    ),
                    participants=replace(
                        config.participants,
                        warp_roles=("one output dot product",) * 4,
                        registers_per_thread=32,
                    ),
                    completion=replace(
                        config.completion,
                        register_fragments="warp FP32 sums are reduced and stored before return",
                    ),
                    evidence=ResourceEvidence(
                        measured=False,
                        source="megabake SM12x batch-one warp GEMV",
                        revision="sm12x-gemv-v1",
                        resources={"threads": 128, "shared_bytes": 0},
                        measurements={},
                    ),
                ),
            )
        wide_mma = tuple(
            replace(
                config,
                name=f"sm12x_mma_16x{columns}x16",
                tiles=replace(config.tiles, cta_shape=(16, columns)),
                evidence=ResourceEvidence(
                    measured=False,
                    source="megabake SM12x wide warp MMA",
                    revision="sm12x-mma-wide-v1",
                    resources={"threads": 128, "shared_bytes": 0},
                    measurements={},
                ),
            )
            for config in mma
            for columns in (128,)
        )
        return CandidateSet(wide_mma + gemv + mma, candidates.rejections)

    def emit(
        self,
        program: Any,
        config: BodyConfig,
        *,
        defer_length_publish: bool = False,
        workers: int | None = None,
    ) -> Any:
        if config.name == "sm12x_gemv_warp4":
            return emit_sm12x_gemv(program, config)
        if config.name == "sm12x_mma_16x128x16":
            return emit_sm12x_wide_mma(program, config)
        return emit_blackwell_gemm(
            program,
            replace(config, provider="blackwell_mma", name="blackwell_mma_16x32x16"),
        )


def emit_sm12x_gemv(program: Any, config: BodyConfig) -> Any:
    """Emit a batch-one GEMV with one output dot product per warp."""
    rows, columns = program.store.shape
    if rows != 1 or config.name != "sm12x_gemv_warp4":
        raise ValueError("SM12x warp GEMV requires a one-row output")
    reduction = next(domain.size for domain in program.iteration_domains if domain.axis == "r0")
    right_index = "k, column" if program.loads[1].axes == ("r0", "o1") else "column, k"
    dtype = "cutlass.BFloat16" if program.store.dtype == "torch.bfloat16" else "cutlass.Float16"
    source = f"""import cutlass
import cutlass.cute as cute

@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    warp = thread // 32
    lane = thread % 32
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        column = cutlass.Int32(work_iterator[tile]) * 4 + warp
        accumulator = cutlass.Float32(0.0)
        if column < {columns}:
            for k in cutlass.range(lane, {reduction}, 32):
                accumulator += cutlass.Float32(tensor_bindings[0][0, k]) * cutlass.Float32(tensor_bindings[1][{right_index}])
            accumulator = cute.arch.warp_reduction_sum(accumulator)
            if lane == 0:
                tensor_bindings[2][0, column] = {dtype}(accumulator)
"""
    from ..generic import GeneratedContractionBody

    return GeneratedContractionBody(
        source=source,
        threads_per_cta=128,
        output_tile_elements=4,
        reduction_chunk=32,
        tile_count=(columns + 3) // 4,
        work_step=1,
        work_iterator_encoding="one four-column output tile ordinal per CTA",
        phases=("output_tile", "warp_dot", "epilogue"),
    )


def emit_sm12x_wide_mma(program: Any, config: BodyConfig) -> Any:
    """Emit four adjacent 16x32 warp-MMA fragments per persistent work tile."""
    tile_columns = config.tiles.cta_shape[1]
    if config.tiles.cta_shape != (16, 128):
        raise ValueError("SM12x wide MMA requires a 16x128 output tile")
    rows, columns = program.store.shape
    reduction = next(domain.size for domain in program.iteration_domains if domain.axis == "r0")
    dtype = "cutlass.BFloat16" if program.store.dtype == "torch.bfloat16" else "cutlass.Float16"
    right_index = "(k, n)" if program.loads[1].axes == ("r0", "o1") else "(n, k)"
    column_tiles = (columns + tile_columns - 1) // tile_columns
    segments = tile_columns // 32
    source = f"""import cutlass
import cutlass.cute as cute

@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, 16)), (1, 4, 1))
    thr_mma = mma.get_slice(thread)
    ca = thr_mma.partition_A(cute.make_identity_tensor((16, 16)))
    cb = thr_mma.partition_B(cute.make_identity_tensor((32, 16)))
    cc = thr_mma.partition_C(cute.make_identity_tensor((16, 32)))
    fa = cute.make_rmem_tensor(ca.shape, {dtype})
    fb = cute.make_rmem_tensor(cb.shape, {dtype})
    fc = cute.make_rmem_tensor(cc.shape, cutlass.Float32)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        tile_id = cutlass.Int32(work_iterator[tile])
        row_base = (tile_id // {column_tiles}) * 16
        tile_column = (tile_id % {column_tiles}) * {tile_columns}
        for segment in cutlass.range_constexpr({segments}):
            col_base = tile_column + segment * 32
            fc.fill(0.0)
            for kbase in cutlass.range(0, {reduction}, 16):
                for i in cutlass.range_constexpr(cute.size(fa)):
                    m = row_base + ca[i][0]
                    k = kbase + ca[i][1]
                    fa[i] = {dtype}(0.0)
                    if m < {rows} and k < {reduction}:
                        fa[i] = tensor_bindings[0][m, k]
                for i in cutlass.range_constexpr(cute.size(fb)):
                    n = col_base + cb[i][0]
                    k = kbase + cb[i][1]
                    fb[i] = {dtype}(0.0)
                    if n < {columns} and k < {reduction}:
                        fb[i] = tensor_bindings[1][{right_index}]
                cute.gemm(mma, fc, fa, fb, fc)
            for i in cutlass.range_constexpr(cute.size(fc)):
                m = row_base + cc[i][0]
                n = col_base + cc[i][1]
                if m < {rows} and n < {columns}:
                    tensor_bindings[2][m, n] = {dtype}(fc[i])
"""
    from ..generic import GeneratedContractionBody

    return GeneratedContractionBody(
        source=source,
        threads_per_cta=128,
        output_tile_elements=16 * tile_columns,
        reduction_chunk=16,
        tile_count=((rows + 15) // 16) * column_tiles,
        work_step=1,
        work_iterator_encoding=f"one row-major 16x{tile_columns} output tile ordinal per CTA",
        phases=("output_tile", "warp_mma", "epilogue"),
    )
