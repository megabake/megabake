#include "../data_types.cuh"
#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

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

__device__ __forceinline__ float apply_epilogue(float v, uint16_t op_type) {
    if (op_type == OP_MATMUL_SILU) {
        return v / (1.0f + __expf(-v));
    } else if (op_type == OP_MATMUL_GELU) {
        return v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
    } else if (op_type == OP_MATMUL_GELU_TANH) {
        float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        return v * 0.5f * (1.0f + tanhf(c));
    }
    return v;
}

// Skinny matmul for M < 128: tiles over N columns across all SMs,
// caches A in shared memory, each thread processes one output column at a time.
#define SKINNY_MAX_M 64
__device__ void matmul_skinny(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    int tile_id, int num_tiles, uint16_t op_type)
{
    int active_tiles = min(num_tiles, (int)gridDim.x);
    int cols_per_tile = (N + active_tiles - 1) / active_tiles;
    int n_start = tile_id * cols_per_tile;
    int n_end = min(n_start + cols_per_tile, N);
    if (n_start >= N) return;

    extern __shared__ char smem[];
    __half* a_smem = (__half*)smem;

    int bk = min(K, 51200 / max(M, 1));
    bk = bk & ~7;
    if (bk < 8) bk = 8;

    for (int n_base = n_start; n_base < n_end; n_base += (int)blockDim.x) {
        int my_n = n_base + (int)threadIdx.x;
        bool valid = (my_n < n_end);

        float acc[SKINNY_MAX_M];
        #pragma unroll
        for (int m = 0; m < SKINNY_MAX_M; m++) acc[m] = 0.0f;

        for (int k_base = 0; k_base < K; k_base += bk) {
            int kc = min(bk, K - k_base);
            int a_total = M * kc;
            for (int i = (int)threadIdx.x; i < a_total; i += (int)blockDim.x) {
                int mr = i / kc;
                int kr = i % kc;
                a_smem[i] = A[(int64_t)mr * K + k_base + kr];
            }
            __syncthreads();

            if (valid) {
                const __half* b_ptr = B + (int64_t)my_n * K + k_base;
                int kv = kc & ~7;
                for (int k = 0; k < kv; k += 8) {
                    float4 bv = *(const float4*)(b_ptr + k);
                    const __half2* bh = (const __half2*)&bv;
                    float2 bf0 = __half22float2(bh[0]);
                    float2 bf1 = __half22float2(bh[1]);
                    float2 bf2 = __half22float2(bh[2]);
                    float2 bf3 = __half22float2(bh[3]);
                    for (int m = 0; m < M && m < SKINNY_MAX_M; m++) {
                        const float4 av = *(const float4*)(a_smem + m * kc + k);
                        const __half2* ah = (const __half2*)&av;
                        float2 af0 = __half22float2(ah[0]);
                        float2 af1 = __half22float2(ah[1]);
                        float2 af2 = __half22float2(ah[2]);
                        float2 af3 = __half22float2(ah[3]);
                        acc[m] += af0.x * bf0.x + af0.y * bf0.y
                                + af1.x * bf1.x + af1.y * bf1.y
                                + af2.x * bf2.x + af2.y * bf2.y
                                + af3.x * bf3.x + af3.y * bf3.y;
                    }
                }
                for (int k = kv; k < kc; k++) {
                    float bval = __half2float(b_ptr[k]);
                    for (int m = 0; m < M && m < SKINNY_MAX_M; m++)
                        acc[m] += __half2float(a_smem[m * kc + k]) * bval;
                }
            }
            __syncthreads();
        }

        if (valid) {
            for (int m = 0; m < M && m < SKINNY_MAX_M; m++)
                C[(int64_t)m * N + my_n] = __float2half(apply_epilogue(acc[m], op_type));
        }
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

// ==================== SM90 WGMMA path ====================

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int STAGES = 3;

// WGMMA: 64x128x16, fp16→fp32, both operands from shared memory (SS)
// 2 warpgroups tiled in M → 128 threads × 2 = 256 threads, BM=128
using MmaAtom_t = MMA_Atom<SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>>;
using TiledMma_t = TiledMMA<MmaAtom_t, Layout<Shape<_2, _1, _1>>>;

// GMMA-compatible 128-byte swizzle (Swizzle<3,4,3>)
using SmemLayoutA_t = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<half_t>{},
    make_shape(Int<BM>{}, Int<BK>{}, Int<STAGES>{})));
using SmemLayoutB_t = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<half_t>{},
    make_shape(Int<BN>{}, Int<BK>{}, Int<STAGES>{})));

// cp.async: 256 threads, K-major access, 128-bit vectorized loads
using G2SCopy_t = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_32, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>>{}));

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

    if (M <= 4) {
        matmul_skinny((const __half*)A, (const __half*)B, (__half*)C,
                      M, N, K, tile_id, task.num_tiles, task.op_type);
        return;
    }

    // --- CuTe WGMMA GEMM (TN layout, transposed B) ---

    int tiles_m = (M + BM - 1) / BM;
    int tiles_n = (N + BN - 1) / BN;
    int total_out_tiles = tiles_m * tiles_n;
    int active_tiles = min((int)task.num_tiles, (int)gridDim.x);
    int per_sm = (total_out_tiles + active_tiles - 1) / active_tiles;
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

    // Shared memory with 128-byte alignment for GMMA descriptors
    extern __shared__ char smem[];
    constexpr int smem_a_size = cosize(SmemLayoutA_t{}) * sizeof(half_t);
    constexpr int smem_a_aligned = (smem_a_size + 127) & ~127;
    half_t* sa_ptr = reinterpret_cast<half_t*>(smem);
    half_t* sb_ptr = reinterpret_cast<half_t*>(smem + smem_a_aligned);
    auto sA = make_tensor(make_smem_ptr(sa_ptr), SmemLayoutA_t{});
    auto sB = make_tensor(make_smem_ptr(sb_ptr), SmemLayoutB_t{});

    TiledMma_t tiled_mma;
    G2SCopy_t  g2s_copy_a;
    G2SCopy_t  g2s_copy_b;
    int idx = threadIdx.x;

    // Position-independent swizzle for cp.async destinations
    Tensor sA_pi = as_position_independent_swizzle_tensor(sA);
    Tensor sB_pi = as_position_independent_swizzle_tensor(sB);

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
        Tensor tAsA = thr_g2s_a.partition_D(sA_pi);
        Tensor tAidA = thr_g2s_a.partition_S(idA);

        ThrCopy thr_g2s_b = g2s_copy_b.get_slice(idx);
        Tensor tBgB = thr_g2s_b.partition_S(gB);
        Tensor tBsB = thr_g2s_b.partition_D(sB_pi);
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

        // --- MMA partitions (GMMA smem descriptors, not register copies) ---
        ThrMMA thr_mma_s = tiled_mma.get_slice(idx);
        Tensor tCsA = thr_mma_s.partition_A(sA);
        Tensor tCsB = thr_mma_s.partition_B(sB);
        Tensor tCgC = thr_mma_s.partition_C(gC);
        Tensor tCrA = thr_mma_s.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma_s.make_fragment_B(tCsB);
        Tensor tCrC = thr_mma_s.make_fragment_C(tCgC);
        clear(tCrC);

        // --- Pipeline ---
        auto K_TILE_MAX = size<3>(tAgA);
        auto K_PIPE_MAX = size<3>(tAsA);

        // Prefetch first STAGES-1 tiles
        CUTE_UNROLL
        for (int p = 0; p < K_PIPE_MAX - 1; ++p) {
            copy_if(g2s_copy_a, pA, tAgA(_, _, _, p), tAsA(_, _, _, p));
            copy_if(g2s_copy_b, pB, tBgB(_, _, _, p), tBsB(_, _, _, p));
            cp_async_fence();
        }

        int k_pipe_read  = 0;
        int k_pipe_write = K_PIPE_MAX - 1;

        // Main K-loop
        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
            int k_tile_next = k_tile + (K_PIPE_MAX - 1);
            k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX - 1 : k_tile_next;

            // Issue cp.async for next tile
            copy_if(g2s_copy_a, pA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe_write));
            copy_if(g2s_copy_b, pB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe_write));
            cp_async_fence();

            ++k_pipe_write;
            k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

            // Wait for current tile
            cp_async_wait<STAGES-2>();
            __syncthreads();

            // WGMMA compute with warpgroup synchronization
            warpgroup_fence_operand(tCrC);
            warpgroup_arrive();
            gemm(tiled_mma, tCrA(_, _, _, k_pipe_read), tCrB(_, _, _, k_pipe_read), tCrC);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(tCrC);

            ++k_pipe_read;
            k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
        }

        // --- Epilogue ---
        Tensor tCidC = thr_mma_s.partition_C(idC);
        const uint16_t op = task.op_type;

        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
            if (elem_less(tCidC(i), make_shape(M, N))) {
                tCgC(i) = half_t(apply_epilogue(tCrC(i), op));
            }
        }

        __syncthreads();
    }
}

#else

// ==================== SM80 path ====================

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 32;
static constexpr int STAGES = 2;

// SM80 MMA: 16x8x16, fp16 in, fp32 out, TN layout (both A and B are K-contiguous)
using MmaAtom_t = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
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

    if (M <= 4) {
        matmul_skinny((const __half*)A, (const __half*)B, (__half*)C,
                      M, N, K, tile_id, task.num_tiles, task.op_type);
        return;
    }

    // --- CuTe tensor-core GEMM (TN layout, transposed B) ---

    int tiles_m = (M + BM - 1) / BM;
    int tiles_n = (N + BN - 1) / BN;
    int total_out_tiles = tiles_m * tiles_n;
    int active_tiles = min((int)task.num_tiles, (int)gridDim.x);
    int per_sm = (total_out_tiles + active_tiles - 1) / active_tiles;
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

        // --- Epilogue: optional activation on fp32 accum, then convert to fp16 ---
        Tensor tCidC = thr_mma_s.partition_C(idC);
        const uint16_t op = task.op_type;

        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
            if (elem_less(tCidC(i), make_shape(M, N))) {
                tCgC(i) = half_t(apply_epilogue(tCrC(i), op));
            }
        }

        __syncthreads();
    }
}

#endif
