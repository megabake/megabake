#include "../data_types.cuh"
#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 32;
static constexpr int STAGES = 2;

// SM80 MMA: 16x8x16, fp16 in, fp32 out, TN layout (both A and B are K-contiguous)
using MmaAtom_t = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
// 2x4 atom grid = 256 threads. Tile<128,128,16> makes each gemm() cover the full
// BM×BN output tile with register-level repetition, which also makes A and B
// fragments 128-bit aligned for LDSM loads.
using TiledMma_t = TiledMMA<MmaAtom_t,
    Layout<Shape<_2, _4, _1>>,
    Tile<Int<BM>, Int<BN>, _16>>;

// Swizzled shared memory
using SmemAtom_t = decltype(composition(
    Swizzle<3, 3, 3>{},
    make_layout(make_shape(Int<8>{}, Int<BK>{}),
                make_stride(Int<BK>{}, Int<1>{}))
));
using SmemLayoutA_t = decltype(tile_to_shape(SmemAtom_t{},
    make_shape(Int<BM>{}, Int<BK>{}, Int<STAGES>{})));
using SmemLayoutB_t = decltype(tile_to_shape(SmemAtom_t{},
    make_shape(Int<BN>{}, Int<BK>{}, Int<STAGES>{})));

// G2S: cp.async 128-bit, 256 threads arranged 64×4
using G2SCopy_t = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<Int<64>, Int<4>>, Stride<Int<4>, Int<1>>>{},
    Layout<Shape<Int<1>,  Int<8>>>{}
));

// S2R: LDSM 128-bit for register loading
using S2RAtomA_t = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
using S2RAtomB_t = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

// Scalar fallback for non-transposed B (rare)
__device__ void matmul_scalar(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K, int b_trans,
    int tile_id, int num_tiles)
{
    const int TM = 16, TN = 16;
    int tn = (N + TN - 1) / TN;
    int tm = (M + TM - 1) / TM;
    int tot = tm * tn;
    int per = (tot + num_tiles - 1) / num_tiles;
    int ts = tile_id * per;
    int te = ts + per < tot ? ts + per : tot;
    for (int t = ts; t < te; t++) {
        int ms = (t / tn) * TM;
        int ns = (t % tn) * TN;
        for (int loc = threadIdx.x; loc < TM * TN; loc += blockDim.x) {
            int gm = ms + loc / TN;
            int gn = ns + loc % TN;
            if (gm < M && gn < N) {
                float acc = 0.0f;
                if (b_trans) {
                    for (int k = 0; k < K; k++)
                        acc += __half2float(A[gm*K+k]) * __half2float(B[gn*K+k]);
                } else {
                    for (int k = 0; k < K; k++)
                        acc += __half2float(A[gm*K+k]) * __half2float(B[k*N+gn]);
                }
                C[gm*N+gn] = __float2half(acc);
            }
        }
    }
}

__device__ void task_matmul(const TaskDesc& task, void** buffers,
                            const int* dyn_dims, int tile_id) {
    half_t*       C = (half_t*)buffers[task.buffer_indices[0]];
    const half_t* A = (const half_t*)buffers[task.buffer_indices[1]];
    const half_t* B = (const half_t*)buffers[task.buffer_indices[2]];

    const int M = task.dimensions[0];
    const int N = task.dimensions[1];
    const int K = task.dimensions[2];
    const int b_transposed = task.strides[0];

    if (!b_transposed) {
        matmul_scalar((const __half*)A, (const __half*)B, (__half*)C,
                      M, N, K, b_transposed, tile_id, task.num_tiles);
        return;
    }

    // --- CuTe tensor-core GEMM (TN layout, transposed B) ---

    int tiles_m = (M + BM - 1) / BM;
    int tiles_n = (N + BN - 1) / BN;
    int total_out_tiles = tiles_m * tiles_n;
    int per_sm = (total_out_tiles + (int)task.num_tiles - 1) / (int)task.num_tiles;
    int my_start = tile_id * per_sm;
    int my_end   = my_start + per_sm;
    if (my_end > total_out_tiles) my_end = total_out_tiles;
    if (my_start >= total_out_tiles) return;

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});

    // A: (M,K) row-major, K-contiguous
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    // B: (N,K) row-major, K-contiguous (transposed weight)
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));
    // C: (M,N) row-major
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    // Identity tensors for boundary predication
    auto cA = make_identity_tensor(make_shape(M, K));
    auto cB = make_identity_tensor(make_shape(N, K));
    auto cC = make_identity_tensor(make_shape(M, N));

    // Shared memory
    extern __shared__ char smem[];
    half_t* sa_ptr = reinterpret_cast<half_t*>(smem);
    half_t* sb_ptr = sa_ptr + cosize(SmemLayoutA_t{});
    auto sA = make_tensor(make_smem_ptr(sa_ptr), SmemLayoutA_t{});
    auto sB = make_tensor(make_smem_ptr(sb_ptr), SmemLayoutB_t{});

    TiledMma_t tiled_mma;
    G2SCopy_t  g2s_copy_a;
    G2SCopy_t  g2s_copy_b;
    int idx = threadIdx.x;

    // S2R copies
    auto s2r_copy_a = make_tiled_copy_A(S2RAtomA_t{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(S2RAtomB_t{}, tiled_mma);

    for (int t = my_start; t < my_end; t++) {
        int tm = t / tiles_n;
        int tn = t % tiles_n;
        auto cta_coord = make_coord(tm, tn, _);

        Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
        Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});
        Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

        Tensor idA = local_tile(cA, cta_tiler, cta_coord, Step<_1, X, _1>{});
        Tensor idB = local_tile(cB, cta_tiler, cta_coord, Step< X, _1, _1>{});
        Tensor idC = local_tile(cC, cta_tiler, cta_coord, Step<_1, _1, X>{});

        // --- G2S partitions ---
        ThrCopy thr_g2s_a = g2s_copy_a.get_slice(idx);
        Tensor tAgA = thr_g2s_a.partition_S(gA);
        Tensor tAsA = thr_g2s_a.partition_D(sA);
        Tensor tAidA = thr_g2s_a.partition_S(idA);

        ThrCopy thr_g2s_b = g2s_copy_b.get_slice(idx);
        Tensor tBgB = thr_g2s_b.partition_S(gB);
        Tensor tBsB = thr_g2s_b.partition_D(sB);
        Tensor tBidB = thr_g2s_b.partition_S(idB);

        // M/N boundary predicates
        auto pA = make_tensor<bool>(make_shape(size<1>(tAidA), Int<1>{}),
                                    make_stride(Int<1>{}, Int<0>{}));
        CUTE_UNROLL
        for (int i = 0; i < size<0>(pA); ++i) {
            pA(i, 0) = get<0>(tAidA(0, i, 0, 0)) < M;
        }

        auto pB = make_tensor<bool>(make_shape(size<1>(tBidB), Int<1>{}),
                                    make_stride(Int<1>{}, Int<0>{}));
        CUTE_UNROLL
        for (int i = 0; i < size<0>(pB); ++i) {
            pB(i, 0) = get<0>(tBidB(0, i, 0, 0)) < N;
        }

        // --- MMA partitions ---
        ThrMMA thr_mma_s = tiled_mma.get_slice(idx);
        Tensor tCgC = thr_mma_s.partition_C(gC);
        Tensor tCrA = thr_mma_s.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB = thr_mma_s.partition_fragment_B(sB(_, _, 0));
        Tensor tCrC = thr_mma_s.make_fragment_C(tCgC);
        clear(tCrC);

        // --- S2R partitions ---
        auto s2r_thr_a = s2r_copy_a.get_slice(idx);
        Tensor tXsA = s2r_thr_a.partition_S(sA);
        Tensor tXrA = s2r_thr_a.retile_D(tCrA);

        auto s2r_thr_b = s2r_copy_b.get_slice(idx);
        Tensor tXsB = s2r_thr_b.partition_S(sB);
        Tensor tXrB = s2r_thr_b.retile_D(tCrB);

        // --- Pipeline ---
        auto K_PIPE_MAX = size<3>(tAsA);
        int k_tile_count = size<3>(tAgA);
        int k_tile_next = 0;

        // Prefetch stages 0..STAGES-2
        CUTE_UNROLL
        for (int p = 0; p < K_PIPE_MAX - 1; ++p) {
            copy_if(g2s_copy_a, pA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, p));
            copy_if(g2s_copy_b, pB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, p));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) ++k_tile_next;
        }

        int smem_pipe_read  = 0;
        int smem_pipe_write = K_PIPE_MAX - 1;
        auto K_BLOCK_MAX = size<2>(tCrA);

        // Prefetch first register tile
        Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
        Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

        if (K_BLOCK_MAX > 1) {
            cp_async_wait<K_PIPE_MAX - 2>();
            __syncthreads();
            copy(S2RAtomA_t{}, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
            copy(S2RAtomB_t{}, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
        }

        // Main K-loop
        CUTE_NO_UNROLL
        while (k_tile_count > -(K_PIPE_MAX - 1)) {
            CUTE_UNROLL
            for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
                if (k_block == K_BLOCK_MAX - 1) {
                    tXsA_p = tXsA(_, _, _, smem_pipe_read);
                    tXsB_p = tXsB(_, _, _, smem_pipe_read);
                    cp_async_wait<K_PIPE_MAX - 2>();
                    __syncthreads();
                }

                auto k_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                copy(S2RAtomA_t{}, tXsA_p(_, _, k_next), tXrA(_, _, k_next));
                copy(S2RAtomB_t{}, tXsB_p(_, _, k_next), tXrB(_, _, k_next));

                if (k_block == 0) {
                    copy_if(g2s_copy_a, pA,
                            tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                    copy_if(g2s_copy_b, pB,
                            tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                    cp_async_fence();
                    --k_tile_count;
                    if (k_tile_count > 0) ++k_tile_next;
                    smem_pipe_write = smem_pipe_read;
                    smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
                }

                gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
            }
        }

        // --- Epilogue: fp32 accum -> fp16, write to global with bounds check ---
        Tensor tCidC = thr_mma_s.partition_C(idC);

        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
            if (elem_less(tCidC(i), make_shape(M, N))) {
                tCgC(i) = half_t(tCrC(i));
            }
        }

        __syncthreads();
    }
}
