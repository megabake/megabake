// V3R-005: direct cuBLASLt FP16 GEMM baseline for Y[M,N] = X[M,K] W[N,K]^T.
#include <cuda_runtime_api.h>
#include <cublasLt.h>

#include <cstdint>
#include <cstdio>
#include <new>

struct LtContext {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t operation{};
  cublasLtMatmulPreference_t preference{};
  cublasLtMatrixLayout_t a{};
  cublasLtMatrixLayout_t b{};
  cublasLtMatrixLayout_t c{};
  cublasLtMatmulAlgo_t algorithm{};
  size_t workspace_bytes{};
};

extern "C" int mb3_cublaslt_create(int64_t m, int64_t n, int64_t k,
                                   void **context, int *algorithm_id,
                                   size_t *workspace_bytes) {
  if (!context || !algorithm_id || !workspace_bytes || m <= 0 || n <= 0 || k <= 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  auto *ctx = new (std::nothrow) LtContext;
  if (!ctx) return static_cast<int>(CUBLAS_STATUS_ALLOC_FAILED);
  int stage = 1;
  cublasStatus_t status = cublasLtCreate(&ctx->handle);
  stage = 2;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescCreate(&ctx->operation, CUBLAS_COMPUTE_32F,
                                     CUDA_R_32F);
  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        ctx->operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
  stage = 3;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        ctx->operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
  // Row-major X/W/output are represented by column-major views of their bytes:
  // W[K,N] is transposed to N[K], X[K,M] stays KxM, and C[N,M] is Y[M,N].
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&ctx->a, CUDA_R_16F, k, n, k);
  stage = 4;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&ctx->b, CUDA_R_16F, k, m, k);
  stage = 5;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&ctx->c, CUDA_R_16F, n, m, n);
  stage = 6;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulPreferenceCreate(&ctx->preference);
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulAlgoGetHeuristic(ctx->handle, ctx->operation,
                                           ctx->a, ctx->b, ctx->c, ctx->c,
                                           ctx->preference, 1, &heuristic, &returned);
  stage = 7;
  if (status == CUBLAS_STATUS_SUCCESS && returned == 0)
    status = CUBLAS_STATUS_NOT_SUPPORTED;
  if (status == CUBLAS_STATUS_SUCCESS) {
    ctx->algorithm = heuristic.algo;
    ctx->workspace_bytes = heuristic.workspaceSize;
    size_t bytes_written = 0;
    stage = 8;
    status = cublasLtMatmulAlgoConfigGetAttribute(
        &ctx->algorithm, CUBLASLT_ALGO_CONFIG_ID, algorithm_id,
        sizeof(*algorithm_id), &bytes_written);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    *workspace_bytes = heuristic.workspaceSize;
    *context = ctx;
    return static_cast<int>(CUBLAS_STATUS_SUCCESS);
  }
  if (ctx->c) cublasLtMatrixLayoutDestroy(ctx->c);
  if (ctx->b) cublasLtMatrixLayoutDestroy(ctx->b);
  if (ctx->a) cublasLtMatrixLayoutDestroy(ctx->a);
  if (ctx->operation) cublasLtMatmulDescDestroy(ctx->operation);
  if (ctx->preference) cublasLtMatmulPreferenceDestroy(ctx->preference);
  if (ctx->handle) cublasLtDestroy(ctx->handle);
  delete ctx;
  std::fprintf(stderr, "cublasLt setup failed at stage %d with status %d\n",
               stage, static_cast<int>(status));
  return static_cast<int>(status);
}

extern "C" int mb3_cublaslt_launch(void *opaque, const void *weight,
                                   const void *x, void *output,
                                   uintptr_t stream_value) {
  auto *ctx = static_cast<LtContext *>(opaque);
  if (!ctx || !weight || !x || !output)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasLtMatmul(
      ctx->handle, ctx->operation, &alpha, weight, ctx->a, x, ctx->b, &beta,
      output, ctx->c, output, ctx->c, &ctx->algorithm, nullptr, 0, stream);
  return static_cast<int>(status);
}

extern "C" void mb3_cublaslt_destroy(void *opaque) {
  auto *ctx = static_cast<LtContext *>(opaque);
  if (!ctx) return;
  if (ctx->c) cublasLtMatrixLayoutDestroy(ctx->c);
  if (ctx->b) cublasLtMatrixLayoutDestroy(ctx->b);
  if (ctx->a) cublasLtMatrixLayoutDestroy(ctx->a);
  if (ctx->operation) cublasLtMatmulDescDestroy(ctx->operation);
  if (ctx->preference) cublasLtMatmulPreferenceDestroy(ctx->preference);
  if (ctx->handle) cublasLtDestroy(ctx->handle);
  delete ctx;
}
