#pragma once
#include <cstdint>

#define OP_MATMUL       0x01
#define OP_ATTENTION    0x02
#define OP_ELEMENTWISE  0x03
#define OP_REDUCE       0x04
#define OP_EMBEDDING    0x05
#define OP_INDEX        0x06
#define OP_COPY         0x07
#define OP_ROPE         0x08
#define OP_MATMUL_SILU        0x09
#define OP_MATMUL_GELU        0x0A
#define OP_FUSED_ELEMENTWISE  0x0B
#define OP_MATMUL_GELU_TANH   0x0C

// Micro-op codes for fused elementwise interpreter
#define UOP_LOAD    0x00
#define UOP_STORE   0x01
#define UOP_ADD     0x02
#define UOP_MUL     0x03
#define UOP_SUB     0x04
#define UOP_DIV     0x05
#define UOP_SILU    0x06
#define UOP_GELU    0x07
#define UOP_RELU    0x08
#define UOP_TANH_U  0x09
#define UOP_NEG     0x0A
#define UOP_EXP     0x0B
#define UOP_SIGMOID 0x0C
#define UOP_RSQRT   0x0D
#define UOP_LOG     0x0E
#define UOP_ABS     0x0F
#define UOP_GELU_TANH 0x10

#define ELEM_ADD          0x00
#define ELEM_MUL          0x01
#define ELEM_SUB          0x02
#define ELEM_DIV          0x03
#define ELEM_SILU         0x04
#define ELEM_GELU         0x05
#define ELEM_RELU         0x06
#define ELEM_SIGMOID      0x07
#define ELEM_TANH_OP      0x08
#define ELEM_EXP          0x09
#define ELEM_LOG          0x0A
#define ELEM_RSQRT        0x0B
#define ELEM_NEG          0x0C
#define ELEM_ABS          0x0D
#define ELEM_CLAMP        0x0E
#define ELEM_WHERE        0x0F
#define ELEM_CAST         0x10
#define ELEM_MASKED_FILL  0x11
#define ELEM_POW          0x12
#define ELEM_COS          0x13
#define ELEM_SIN          0x14
#define ELEM_GELU_TANH    0x15

#define COPY_SIMPLE       0x00
#define COPY_CAT          0x01

#define REDUCE_SUM        0x00
#define REDUCE_MEAN       0x01
#define REDUCE_MAX        0x02
#define REDUCE_SOFTMAX    0x03
#define REDUCE_RMSNORM    0x04
#define REDUCE_LAYERNORM  0x05
#define REDUCE_ARGMAX     0x06

#define INDEX_GATHER        0x00
#define INDEX_SCATTER       0x01
#define INDEX_INDEX_SELECT  0x02
#define INDEX_INDEX_PUT     0x03

#define DTYPE_FP16    0x00
#define DTYPE_BF16    0x01
#define DTYPE_FP32    0x02
#define DTYPE_FP8E4M3 0x03
#define DTYPE_INT8    0x04
#define DTYPE_INT32   0x05
#define DTYPE_INT64   0x06
#define DTYPE_BOOL    0x07

struct __align__(16) TaskDesc {
    uint16_t op_type;
    uint16_t op_code;
    uint32_t num_tiles;
    uint32_t buffer_indices[8];
    uint32_t dimensions[8];
    int32_t  strides[8];
};

struct __align__(8) BufferDesc {
    uint64_t offset;
    uint64_t size;
    uint16_t dtype;
    uint16_t _pad0;
    uint32_t _pad1;
};

struct __align__(4) WeightMapping {
    uint32_t buffer_index;
    uint32_t key_offset;
    uint16_t key_length;
    uint16_t _pad;
};

struct ScheduleHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_tasks;
    uint32_t num_buffers;
    uint64_t workspace_bytes;
    uint32_t num_weight_mappings;
    uint32_t batch_min;
    uint32_t batch_max;
    uint32_t seq_min;
    uint32_t seq_max;
    uint32_t sm_version;
    uint16_t compute_dtype;
    uint16_t _padding;
};
