from dataclasses import dataclass, field
from enum import IntEnum
import struct


class OpType(IntEnum):
    MATMUL             = 0x01
    ATTENTION          = 0x02
    ELEMENTWISE        = 0x03
    REDUCE             = 0x04
    EMBEDDING          = 0x05
    INDEX              = 0x06
    COPY               = 0x07
    ROPE               = 0x08
    FUSED_ELEMENTWISE  = 0x0B


EPILOGUE_SILU      = 0x01
EPILOGUE_GELU      = 0x02
EPILOGUE_GELU_TANH = 0x04
EPILOGUE_BIAS      = 0x08
EPILOGUE_RESIDUAL  = 0x10


class ElemCode(IntEnum):
    ADD         = 0x00
    MUL         = 0x01
    SUB         = 0x02
    DIV         = 0x03
    SILU        = 0x04
    GELU        = 0x05
    RELU        = 0x06
    SIGMOID     = 0x07
    TANH        = 0x08
    EXP         = 0x09
    LOG         = 0x0A
    RSQRT       = 0x0B
    NEG         = 0x0C
    ABS         = 0x0D
    CLAMP       = 0x0E
    WHERE       = 0x0F
    CAST        = 0x10
    MASKED_FILL = 0x11
    POW         = 0x12
    COS         = 0x13
    SIN         = 0x14
    GELU_TANH   = 0x15


class UopCode(IntEnum):
    LOAD    = 0x00
    STORE   = 0x01
    ADD     = 0x02
    MUL     = 0x03
    SUB     = 0x04
    DIV     = 0x05
    SILU    = 0x06
    GELU    = 0x07
    RELU    = 0x08
    TANH    = 0x09
    NEG     = 0x0A
    EXP     = 0x0B
    SIGMOID = 0x0C
    RSQRT   = 0x0D
    LOG     = 0x0E
    ABS     = 0x0F
    GELU_TANH      = 0x10
    LOAD_BROADCAST = 0x11
    REDUCE_SUM     = 0x12
    REDUCE_MAX     = 0x13
    REDUCE_MEAN    = 0x14


def pack_uop(opcode: int, dst: int = 0, src1: int = 0, src2: int = 0) -> int:
    return (opcode & 0xFF) | ((dst & 0xF) << 8) | ((src1 & 0xF) << 12) | ((src2 & 0xF) << 16)


class ReduceCode(IntEnum):
    SUM       = 0x00
    MEAN      = 0x01
    MAX       = 0x02
    SOFTMAX   = 0x03
    RMSNORM   = 0x04
    LAYERNORM = 0x05
    ARGMAX    = 0x06


class IndexCode(IntEnum):
    GATHER       = 0x00
    SCATTER      = 0x01
    INDEX_SELECT = 0x02
    INDEX_PUT    = 0x03


class DTypeCode(IntEnum):
    FP16    = 0x00
    BF16    = 0x01
    FP32    = 0x02
    FP8E4M3 = 0x03
    INT8    = 0x04
    INT32   = 0x05
    INT64   = 0x06
    BOOL    = 0x07


UNUSED_BUFFER = 0xFFFFFFFF


@dataclass
class TaskDesc:
    op_type: int
    op_code: int = 0
    num_tiles: int = 0
    buffer_indices: list[int] = field(
        default_factory=lambda: [UNUSED_BUFFER] * 8
    )
    dimensions: list[int] = field(default_factory=lambda: [0] * 8)
    strides: list[int] = field(default_factory=lambda: [0] * 8)

    STRUCT_FORMAT = "<HHI 8I 8I 8i 8x"
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)  # 112 (matches __align__(16) C struct)

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.STRUCT_FORMAT,
            self.op_type, self.op_code, self.num_tiles,
            *self.buffer_indices,
            *self.dimensions,
            *self.strides,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TaskDesc":
        vals = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(
            op_type=vals[0], op_code=vals[1], num_tiles=vals[2],
            buffer_indices=list(vals[3:11]),
            dimensions=list(vals[11:19]),
            strides=list(vals[19:27]),
        )


@dataclass
class BufferDesc:
    offset: int
    size: int
    dtype: int
    buffer_id: int = 0

    STRUCT_FORMAT = "<QQHxx I"
    STRUCT_SIZE = 24

    def to_bytes(self) -> bytes:
        return struct.pack(self.STRUCT_FORMAT,
                           self.offset, self.size, self.dtype, self.buffer_id)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BufferDesc":
        offset, size, dtype, buffer_id = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(offset=offset, size=size, dtype=dtype, buffer_id=buffer_id)


@dataclass
class WeightMapping:
    buffer_index: int
    key_offset: int
    key_length: int

    STRUCT_FORMAT = "<IIHxx"
    STRUCT_SIZE = 12

    def to_bytes(self) -> bytes:
        return struct.pack(self.STRUCT_FORMAT,
                           self.buffer_index, self.key_offset, self.key_length)

    @classmethod
    def from_bytes(cls, data: bytes) -> "WeightMapping":
        buf_idx, key_off, key_len = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(buffer_index=buf_idx, key_offset=key_off, key_length=key_len)


SCHEDULE_MAGIC = 0x4D454741  # "MEGA"
SCHEDULE_VERSION = 1


@dataclass
class ScheduleHeader:
    magic: int = SCHEDULE_MAGIC
    version: int = SCHEDULE_VERSION
    num_tasks: int = 0
    num_buffers: int = 0
    workspace_bytes: int = 0
    num_weight_mappings: int = 0
    batch_min: int = 0
    batch_max: int = 0
    seq_min: int = 0
    seq_max: int = 0
    sm_version: int = 0
    compute_dtype: int = 0

    STRUCT_FORMAT = "<IIIIQ IIIII IHxx"
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)  # 52

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.STRUCT_FORMAT,
            self.magic, self.version, self.num_tasks, self.num_buffers,
            self.workspace_bytes, self.num_weight_mappings,
            self.batch_min, self.batch_max, self.seq_min, self.seq_max,
            self.sm_version, self.compute_dtype,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "ScheduleHeader":
        vals = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(
            magic=vals[0], version=vals[1], num_tasks=vals[2],
            num_buffers=vals[3], workspace_bytes=vals[4],
            num_weight_mappings=vals[5],
            batch_min=vals[6], batch_max=vals[7],
            seq_min=vals[8], seq_max=vals[9],
            sm_version=vals[10], compute_dtype=vals[11],
        )
