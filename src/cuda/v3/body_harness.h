#pragma once

#include <stdint.h>

typedef struct Mb3Contraction {
  const void *x;
  const void *weight;
  const void *bias;
  void *output;
  int64_t m, n, k;
  int64_t x_m_stride, x_k_stride;
  int64_t weight_n_stride, weight_k_stride;
  float alpha, beta;
  int input_dtype;  // 0: fp16, 1: bf16
} Mb3Contraction;

typedef struct Mb3Resources {
  int device_sms;
  int cooperative_launch;
  int active_ctas_per_sm;
  int resident_ctas;
  int registers_per_thread;
  int static_shared_bytes;
  int local_bytes;
} Mb3Resources;

#ifdef __cplusplus
extern "C" {
#endif

int mb3_owner_profile(int strategy, int warps_per_cta, int vector_width,
                      int mainloop_depth, int input_dtype,
                      Mb3Resources *resources);
int mb3_standalone_profile(int strategy, int warps_per_cta, int vector_width,
                           int mainloop_depth, int input_dtype,
                           Mb3Resources *resources);
int mb3_validate_owner_grid(int strategy, int warps_per_cta,
                            int vector_width, int mainloop_depth, int input_dtype,
                            int grid_ctas);
int mb3_launch(const Mb3Contraction *problem, int strategy,
               int warps_per_cta, int vector_width, int mainloop_depth,
               int grid_ctas,
               uintptr_t stream);
// Harness path for a grid already admitted against this exact compiled entry.
int mb3_launch_admitted(const Mb3Contraction *problem, int strategy,
                        int warps_per_cta, int vector_width,
                        int mainloop_depth, int grid_ctas,
                        uintptr_t stream);
int mb3_capture(const Mb3Contraction *problem, int strategy,
                int warps_per_cta, int vector_width, int mainloop_depth,
                int grid_ctas,
                int launches, uintptr_t stream, void **graph_exec);
int mb3_graph_launch(void *graph_exec, uintptr_t stream);
int mb3_graph_destroy(void *graph_exec);
const char *mb3_error_string(int code);

#ifdef __cplusplus
}
#endif
