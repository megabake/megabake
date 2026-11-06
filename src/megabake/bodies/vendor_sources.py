"""Pinned upstream references for device bodies and their local adaptations."""

from pathlib import Path

CUDA = Path(__file__).resolve().parent / "cuda"


def references():
    hopper = CUDA / "sm90"
    hopper_upstream = hopper / "cutlass_upstream"
    metadata = (hopper_upstream / "UPSTREAM").read_text().splitlines()
    name, sha256 = next(
        line.partition(":")[2].split() for line in metadata if line.partition(":")[0] == "sha256"
    )
    blackwell = CUDA / "sm100"
    return (
        {
            "original": hopper_upstream / name,
            "adaptation": hopper / "cutlass_gemm.py",
            "sha256": sha256,
            "header": b"",
        },
        {
            "original": blackwell / "cutlass_upstream" / "fp16_gemm_0.py",
            "adaptation": blackwell / "tcgen_gemm.py",
            "sha256": "a15471df9a2a9c80a6a772416210ea256a58b142aa4b6358ca01851f6f02362a",
            "header": b"# ruff: noqa\n# fmt: off\n",
        },
        {
            "original": blackwell / "cutlass_upstream" / "dense_gemm_persistent.py",
            "adaptation": blackwell / "cutlass_gemm.py",
            "sha256": "4f3d5fa8c6032c2190933cb43c71d358bb200225fe7c0ec0b54014a9a6235362",
            "header": b"# ruff: noqa\n# fmt: off\n",
        },
    )
