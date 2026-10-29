"""Blackwell GEMM bodies, including experimental SM100 short-row paths."""
from __future__ import annotations

from dataclasses import replace
import os
import re

from ....graph import Region
from ....lowering import Accumulator, Binary, Cast, Constant, Load, TensorProgram
from ..generic import GeneratedContractionBody
from ...interface import (
    BodyConfig, CandidateSet, ParticipantContract, ProviderRejection, ResourceEvidence,
    SelectionPolicy, TargetFacts, TileContract, validate_body_config,
)

BLACKWELL_ARCHITECTURES = frozenset({"sm100", "sm103", "sm120", "sm121"})


def _sm100_tma_candidate(program, base, target, policy, paired, fused=False):
    """Return the experimental short-row body when its contracts hold."""
    mode_enabled = (policy.mode == "prefill" or (
        policy.mode == "decode" and os.environ.get("MEGABAKE_SM100_TCGEN_DECODE", "1") != "0"
    ))
    if (paired or target.architecture != "sm100" or not target.supports_tma
            or os.environ.get("MEGABAKE_SM100_TCGEN") != "1"
            or not policy.allow_tma_gemm or not policy.table_launch or not mode_enabled
            or program.store.dtype != "torch.bfloat16"):
        return None
    rows, columns = program.store.shape
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    left, right = program.loads[:2]
    left_strides = dict(zip(left.axes, left.strides, strict=True))
    right_strides = dict(zip(right.axes, right.strides, strict=True))
    tile_setting = os.environ.get("MEGABAKE_SM100_TCGEN_TILE", "128")
    # Some model projections are multiples of 64 but not 128. Allow those
    # phases to use the tensor-core path while preserving 128-feature tiles
    # for the other projections. This is a shape-based choice, not tuning.
    feature_tile = 64 if tile_setting == "64" or (
        tile_setting == "auto" and columns % 128 != 0
    ) else 128
    from .tuning import choice_for_shape
    measured = choice_for_shape(target, rows, columns, reduction)
    if measured is not None:
        feature_tile = measured[0]
    token_setting = os.environ.get("MEGABAKE_SM100_TOKEN_TILE", "8")
    token_tile = int(token_setting) if token_setting in {"8", "16", "32"} else 8
    row_setting = os.environ.get("MEGABAKE_SM100_TCGEN_MAX_ROWS", "32")
    max_rows = int(row_setting) if row_setting in {"32", "64", "128"} else 32
    if (not 1 <= rows <= max_rows or columns < 128 or columns % feature_tile != 0
            or reduction % 64 != 0
            or left_strides != {"o0": reduction, "r0": 1}
            or right_strides != {"r0": 1, "o1": reduction}
            or program.store.strides != (columns, 1)):
        return None
    split_setting = os.environ.get("MEGABAKE_SM100_SPLIT_K", "1")
    split_limit = int(split_setting) if split_setting in {"2", "4", "8"} else 1
    output_tiles = ((rows + token_tile - 1) // token_tile) * (columns // feature_tile)
    # Do not create a second wave of split tiles. Large projections such as
    # the language-model head already have enough parallelism without split-K.
    splits = 1
    for count in (2, 4, 8):
        if (count <= split_limit and reduction % (64 * count) == 0
                and output_tiles * count <= target.multiprocessor_count):
            splits = count
    if measured is not None:
        splits = measured[1]
    from ...catalog import _layouts
    return replace(
        base,
        provider="sm100_tma",
        name=(f"sm100_tma_swapab_{token_tile}x{feature_tile}"
              + (f"_k{splits}" if splits > 1 else "")
              + ("_epilogue" if fused else "")),
        layouts=_layouts(program, alignment_bytes=16),
        tiles=TileContract(
            cta_shape=(token_tile, feature_tile), cluster_shape=(1, 1),
            work_partition=f"persistent range of transposed {feature_tile}x{token_tile} output tiles",
            mma_family="tcgen05", reduction_tile=(64,),
        ),
        participants=ParticipantContract(
            participating_threads=128, waiting_threads=128,
            warp_roles=("tma_mma", "epilogue", "epilogue", "epilogue"),
            registers_per_thread=128,
        ),
        pipeline=replace(
            base.pipeline, stage_count=4,
            barriers=("tma_umma", "umma_epilogue"),
            async_operations=("tma_load", "tcgen05_mma"),
            descriptors={"A": "TMA weight", "B": "TMA activation"},
            drain_rule="wait for the TMEM result and free the allocation before return",
        ),
        scratch=replace(base.scratch, size_bytes=73_728 if token_tile == 8 else 86_016,
                        alignment_bytes=128),
        completion=replace(
            base.completion,
            register_fragments="TMEM accumulators are copied to registers and stored",
        ),
        evidence=ResourceEvidence(
            measured=False, source="experimental SM100 TMA/tcgen05 pipeline",
            revision="cutlass-4.8.0-tutorial", resources={
                "threads": 128, "shared_bytes": 73_728 if token_tile == 8 else 86_016},
            measurements={},
        ),
    )


class BlackwellGemmProvider:
    provider_name = "blackwell_mma"

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
        paired = bool(region.semantic_facts.get("paired"))
        if paired and os.environ.get("MEGABAKE_SM100_PAIRED_MATVEC") != "1":
            return reject("experimental paired matvec is disabled")
        if (program.family != "contraction" or (len(program.loads) != 3 if paired else len(program.loads) < 2)
                or len(program.store.shape) != 2 or program.reduction_axes != ("r0",)):
            return reject("provider supports plain or paired 2-D GEMM only")
        if any(domain.size < 1 for domain in program.iteration_domains):
            return reject("provider requires nonempty GEMM dimensions")
        dtype = program.store.dtype
        if dtype not in {"torch.float16", "torch.bfloat16"} or any(
            load.dtype != dtype for load in program.loads[:3 if paired else 2]
        ):
            return reject("provider requires matching FP16 or BF16 operands and output")
        left, right = program.loads[:2]
        if left.axes != ("o0", "r0") or right.axes not in {("r0", "o1"), ("o1", "r0")}:
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
                or program.combine != Binary("add", acc, product, "torch.float32")):
            return reject("provider requires an unscaled FP32 accumulation and result cast")
        fused = not paired and (len(program.loads) != 2
                                or program.post != Cast(acc, dtype, "round_nearest_even"))
        if fused:
            if os.environ.get("MEGABAKE_SM100_EPILOGUE") != "1":
                return reject("experimental SM100 fused epilogue is disabled")
            from ..sm90.gemm import _epilogue_rejection
            reason = _epilogue_rejection(program, replace(policy, table_gemm_swap_ab=True))
            if reason is not None:
                return reject(reason)
        if paired:
            from ..generic import UnsupportedMapBody, _emit_expression
            second = program.loads[2]
            if (second.axes, second.strides, second.dtype, second.source_shape) != (
                right.axes, right.strides, right.dtype, right.source_shape
            ):
                return reject("paired weights must have the same shape and layout")
            if program.post is None:
                return reject("paired GEMM requires an epilogue")
            try:
                expression = _emit_expression(program.post, {}, {"o0": "0", "o1": "0"}, ())
            except UnsupportedMapBody:
                return reject("paired GEMM epilogue is unsupported")
            accumulator_indices = {
                int(match.group(1) or 0)
                for match in re.finditer(r"\baccumulator(?:_(\d+))?\b", expression.code)
            }
            if accumulator_indices != {0, 1}:
                return reject("paired epilogue must consume its two accumulators")
            output_shape = region.semantic_facts.get("gemm_epilogue_output_shape", program.store.shape)
            if (not output_shape or output_shape[-1] != program.store.shape[1]
                    or any(size != 1 for size in output_shape[:-1])):
                return reject("paired matvec output must be a single contiguous row")
        from ...catalog import _generic_config
        base = _generic_config(region, program)
        mma = replace(
            base,
            provider=self.provider_name,
            name="blackwell_mma_16x32x16",
            tiles=TileContract(
                cta_shape=(16, 32), cluster_shape=(1, 1),
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
                revision="blackwell-mma-v1", resources={"threads": 128, "shared_bytes": 0},
                measurements={},
            ),
        )
        tma = _sm100_tma_candidate(program, base, target, policy, paired, fused)
        if fused:
            return CandidateSet((tma,), ()) if tma is not None else reject(
                "SM100 fused epilogues require the compatible TMA body"
            )
        # Warp reduction changes BF16 accumulation order and can change greedy
        # tokens. Keep it an explicit experiment until wider model validation.
        if not paired and os.environ.get("MEGABAKE_SM100_MATVEC") != "1":
            return CandidateSet((tma, mma) if tma is not None else (mma,), ())
        # A single decode token leaves 15 of 16 MMA rows empty.  On SM100,
        # give each warp several output columns and reduce over K instead.
        # This path is restricted to contiguous K weights and has been
        # exercised on SM100; other Blackwell targets keep the existing body.
        contiguous_k = right.strides[right.axes.index("r0")] == 1
        rows = program.store.shape[0]
        if (target.architecture == "sm100" and 1 <= rows <= 4
                and (not paired or rows == 1)
                and program.store.shape[1] >= 128 and contiguous_k):
            name = ("blackwell_matvec_pair_1x16" if paired else
                    "blackwell_matvec_1x16" if rows == 1 else f"blackwell_smallrows_{rows}x16")
            matvec = replace(
                base,
                provider=self.provider_name,
                name=name,
                tiles=TileContract(
                    cta_shape=(rows, 16), cluster_shape=(1, 1),
                    work_partition="four warps, each reducing four output columns over K",
                    mma_family=None, reduction_tile=(32,),
                ),
                participants=ParticipantContract(
                    participating_threads=128, waiting_threads=128,
                    warp_roles=("matvec",) * 4, registers_per_thread=64,
                ),
                completion=replace(
                    base.completion,
                    register_fragments="warp-reduced FP32 accumulators are cast and stored before return",
                ),
                evidence=ResourceEvidence(
                    measured=False, source="SM100 warp-parallel small-row contraction",
                    revision="sm100-matvec-v1", resources={
                        "threads": 128, "shared_bytes": 0,
                        "output_shape": output_shape if paired else program.store.shape,
                    },
                    measurements={},
                ),
            )
            if tma is not None:
                return CandidateSet((tma, matvec, mma), ())
            return CandidateSet((matvec,) if paired else (matvec, mma), ())
        if tma is not None:
            return CandidateSet((tma, mma), ())
        if paired:
            return reject("paired GEMM is supported only by SM100 single-row matvec")
        return CandidateSet((mma,), ())


def emit_blackwell_gemm(
    program: TensorProgram, config: BodyConfig
) -> GeneratedContractionBody:
    """Emit a synchronous 16x32 CTA tile with predicated M/N/K tails."""
    validate_body_config(config)
    if config.provider == "blackwell_mma" and config.name.startswith("blackwell_smallrows_"):
        return _emit_blackwell_smallrows(program, config)
    if config.provider == "blackwell_mma" and config.name in {
        "blackwell_matvec_1x16", "blackwell_matvec_pair_1x16"
    }:
        return _emit_blackwell_matvec(program, config)
    if config.provider != "blackwell_mma" or config.tiles.cta_shape != (16, 32):
        raise ValueError("Blackwell GEMM requires the 16x32 warp MMA body contract")
    rows, columns = program.store.shape
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    dtype = "cutlass.BFloat16" if program.store.dtype == "torch.bfloat16" else "cutlass.Float16"
    right_index = "(k, n)" if program.loads[1].axes == ("r0", "o1") else "(n, k)"
    ntiles = (columns + 31) // 32
    source = f'''import cutlass
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
        row_base = (tile_id // {ntiles}) * 16
        col_base = (tile_id % {ntiles}) * 32
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
'''
    return GeneratedContractionBody(
        source=source, threads_per_cta=128, output_tile_elements=512,
        reduction_chunk=16, tile_count=((rows + 15) // 16) * ntiles,
        work_iterator_encoding="one row-major 16x32 output tile ordinal per CTA",
        phases=("output_tile", "warp_mma", "epilogue"),
    )


def _emit_blackwell_matvec(
    program: TensorProgram, config: BodyConfig
) -> GeneratedContractionBody:
    """One warp reduces K for four columns, with FP32 accumulation."""
    rows, columns = program.store.shape
    if rows != 1 or config.tiles.cta_shape != (1, 16):
        raise ValueError("Blackwell matvec requires one row and a 1x16 tile")
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    dtype = "cutlass.BFloat16" if program.store.dtype == "torch.bfloat16" else "cutlass.Float16"
    paired = config.name == "blackwell_matvec_pair_1x16"
    transposed = program.loads[1].axes == ("r0", "o1")
    bindings = 4 if paired else 3
    output_shape = config.evidence.resources.get("output_shape", program.store.shape)
    cols = "\n".join(f"        col{i} = tile_id * 16 + warp * 4 + {i}" for i in range(4))
    init = "\n".join(
        f"        acc{i} = cutlass.Float32(0.0)"
        + (f"\n        other{i} = cutlass.Float32(0.0)" if paired else "")
        for i in range(4)
    )
    accum = "\n".join(
        f"            if kk < {reduction} and col{i} < {columns}:\n"
        f"                acc{i} = acc{i} + x * tensor_bindings[1]["
        f"{'kk, col' + str(i) if transposed else 'col' + str(i) + ', kk'}].to(cutlass.Float32)"
        + (f"\n                other{i} = other{i} + x * tensor_bindings[2]["
           f"{'kk, col' + str(i) if transposed else 'col' + str(i) + ', kk'}].to(cutlass.Float32)"
           if paired else "")
        for i in range(4)
    )
    stores = []
    for i in range(4):
        stores.append(f"        sum{i} = cute.arch.warp_reduction_sum(acc{i})")
        if paired:
            from ..generic import _cast_expression, _emit_expression
            stores.append(f"        other_sum{i} = cute.arch.warp_reduction_sum(other{i})")
            result = _cast_expression(
                _emit_expression(program.post, {}, {"o0": "0", "o1": f"col{i}"}, ()),
                program.store.dtype,
            ).code.replace("accumulator_1", f"other_sum{i}").replace("accumulator", f"sum{i}")
        else:
            result = f"{dtype}(sum{i})"
        stores.append(f"        if lane == 0 and col{i} < {columns}:")
        coordinates = ", ".join(["0"] * (len(output_shape) - 1) + [f"col{i}"])
        stores.append(f"            tensor_bindings[{bindings - 1}][{coordinates}] = {result}")
    stores = "\n".join(stores)
    source = f'''import cutlass
import cutlass.cute as cute

@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{', '.join(['cute.Tensor'] * bindings)}],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    lane = thread % 32
    warp = thread // 32
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        tile_id = cutlass.Int32(work_iterator[tile])
{cols}
{init}
        for kk_base in cutlass.range(0, {reduction}, 32):
            kk = kk_base + lane
            x = cutlass.Float32(0.0)
            if kk < {reduction}:
                x = tensor_bindings[0][0, kk].to(cutlass.Float32)
{accum}
{stores}
'''
    return GeneratedContractionBody(
        source=source, threads_per_cta=128, output_tile_elements=16,
        reduction_chunk=32, tile_count=(columns + 15) // 16,
        work_iterator_encoding="one row-major 1x16 output tile ordinal per CTA",
        phases=("output_tile", "warp_reduce", "epilogue"),
    )


def _emit_blackwell_smallrows(
    program: TensorProgram, config: BodyConfig
) -> GeneratedContractionBody:
    """Reuse each weight across two to four rows while warps reduce K."""
    rows, columns = program.store.shape
    if rows not in {2, 3, 4} or config.tiles.cta_shape != (rows, 16):
        raise ValueError("Blackwell small-row body requires two to four rows")
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    dtype = "cutlass.BFloat16" if program.store.dtype == "torch.bfloat16" else "cutlass.Float16"
    transposed = program.loads[1].axes == ("r0", "o1")
    cols = "\n".join(f"        col{i} = tile_id * 16 + warp * 4 + {i}" for i in range(4))
    init = "\n".join(
        f"        acc{row}_{col} = cutlass.Float32(0.0)"
        for row in range(rows) for col in range(4)
    )
    inputs = "\n".join(
        f"                x{row} = tensor_bindings[0][{row}, kk].to(cutlass.Float32)"
        for row in range(rows)
    )
    weights = "\n".join(
        f"                w{col} = cutlass.Float32(0.0)\n"
        f"                if col{col} < {columns}:\n"
        f"                    w{col} = tensor_bindings[1]["
        f"{'kk, col' + str(col) if transposed else 'col' + str(col) + ', kk'}].to(cutlass.Float32)"
        for col in range(4)
    )
    products = "\n".join(
        f"                acc{row}_{col} = acc{row}_{col} + x{row} * w{col}"
        for row in range(rows) for col in range(4)
    )
    stores = "\n".join(
        f"        sum{row}_{col} = cute.arch.warp_reduction_sum(acc{row}_{col})\n"
        f"        if lane == 0 and col{col} < {columns}:\n"
        f"            tensor_bindings[2][{row}, col{col}] = {dtype}(sum{row}_{col})"
        for row in range(rows) for col in range(4)
    )
    source = f'''import cutlass
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
    lane = thread % 32
    warp = thread // 32
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        tile_id = cutlass.Int32(work_iterator[tile])
{cols}
{init}
        for kk_base in cutlass.range(0, {reduction}, 32):
            kk = kk_base + lane
            if kk < {reduction}:
{inputs}
{weights}
{products}
{stores}
'''
    return GeneratedContractionBody(
        source=source, threads_per_cta=128, output_tile_elements=rows * 16,
        reduction_chunk=32, tile_count=(columns + 15) // 16,
        work_iterator_encoding=f"one row-major {rows}x16 output tile ordinal per CTA",
        phases=("output_tile", "warp_reduce", "epilogue"),
    )
