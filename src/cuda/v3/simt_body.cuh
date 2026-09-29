// Callable warp-per-output SIMT body used by the V3 schedule registry.
// The including translation unit provides Mb3Contraction and CUDA FP16 types.

template <int VectorWidth>
__device__ __forceinline__ float simt_scalar_dot(
    const Mb3Contraction &p, int64_t row, int64_t col, int lane) {
  const auto *x = static_cast<const __half *>(p.x);
  const auto *w = static_cast<const __half *>(p.weight);
  float acc = 0.0f;
  for (int64_t base = 0; base < p.k; base += 32 * VectorWidth) {
#pragma unroll
    for (int item = 0; item < VectorWidth; ++item) {
      const int64_t k = base + lane + item * 32;
      if (k < p.k) {
        acc = fmaf(__half2float(x[row * p.x_m_stride + k * p.x_k_stride]),
                   __half2float(w[col * p.weight_n_stride + k * p.weight_k_stride]),
                   acc);
      }
    }
  }
  return acc;
}

template <int FixedK>
__device__ __forceinline__ void simt_half2_accumulate(
    const Mb3Contraction &p, const __half *x, const __half *w, int64_t pair,
    float &even_acc, float &odd_acc) {
  bool valid;
  if constexpr (FixedK > 0)
    valid = pair < FixedK / 2;
  else
    valid = pair < p.k / 2;
  if (valid) {
    const int64_t k = pair * 2;
    const float2 xv = __half22float2(
        *reinterpret_cast<const __half2 *>(x + k));
    const float2 wv = __half22float2(
        *reinterpret_cast<const __half2 *>(w + k));
    even_acc = fmaf(xv.x, wv.x, even_acc);
    odd_acc = fmaf(xv.y, wv.y, odd_acc);
  }
}

template <int FixedK, int VectorWidth>
__device__ __forceinline__ float simt_half2_dot(
    const Mb3Contraction &p, int64_t row, int64_t col, int lane) {
  const auto *x = static_cast<const __half *>(p.x) + row * p.x_m_stride;
  const auto *w = static_cast<const __half *>(p.weight) + col * p.weight_n_stride;
  float even_acc = 0.0f;
  float odd_acc = 0.0f;
  if constexpr (FixedK > 0) {
    constexpr int kPairs = FixedK / 2;
    constexpr int kIterations = (kPairs + 32 * VectorWidth - 1) /
                                (32 * VectorWidth);
#pragma unroll
    for (int iter = 0; iter < kIterations; ++iter) {
#pragma unroll
      for (int item = 0; item < VectorWidth; ++item)
        simt_half2_accumulate<FixedK>(
            p, x, w, iter * 32 * VectorWidth + lane + item * 32,
            even_acc, odd_acc);
    }
  } else {
    for (int64_t base = 0; base < p.k / 2; base += 32 * VectorWidth) {
#pragma unroll
      for (int item = 0; item < VectorWidth; ++item)
        simt_half2_accumulate<FixedK>(
            p, x, w, base + lane + item * 32, even_acc, odd_acc);
    }
  }
  return even_acc + odd_acc;
}

template <int VectorWidth>
__device__ __forceinline__ void simt_output(const Mb3Contraction &p,
                                             int64_t output_index,
                                             int lane) {
  const int64_t count = p.m * p.n;
  if (output_index >= count) return;  // warp-uniform tail
  int64_t row;
  int64_t col;
  if (p.m == 1) {
    row = 0;
    col = output_index;
  } else {
    row = output_index / p.n;
    col = output_index - row * p.n;
  }
  const auto *x = static_cast<const __half *>(p.x);
  const auto *w = static_cast<const __half *>(p.weight);
  const auto *bias = static_cast<const __half *>(p.bias);
  float acc;
  if constexpr (VectorWidth == 4) {
    const auto *x_row = x + row * p.x_m_stride;
    const auto *w_row = w + col * p.weight_n_stride;
    const bool half2_legal = p.x_k_stride == 1 && p.weight_k_stride == 1 &&
        (p.k & 1) == 0 &&
        (reinterpret_cast<uintptr_t>(x_row) & (alignof(__half2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(w_row) & (alignof(__half2) - 1)) == 0;
    if (half2_legal && p.k == 576)
      acc = simt_half2_dot<576, VectorWidth>(p, row, col, lane);
    else if (half2_legal && p.k == 1536)
      acc = simt_half2_dot<1536, VectorWidth>(p, row, col, lane);
    else if (half2_legal)
      acc = simt_half2_dot<0, VectorWidth>(p, row, col, lane);
    else
      acc = simt_scalar_dot<VectorWidth>(p, row, col, lane);
  } else {
    acc = simt_scalar_dot<VectorWidth>(p, row, col, lane);
  }
  for (int offset = 16; offset; offset >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  if (lane == 0) {
    const float c = bias ? __half2float(bias[output_index]) : 0.0f;
    static_cast<__half *>(p.output)[output_index] =
        __float2half_rn(p.alpha * acc + p.beta * c);
  }
}
