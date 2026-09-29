import subprocess

import pytest

from megabake.v3.backends.cuda.bodies.registry import query_body_tactics
from tests.test_v3.cpu.test_body_registry import _linear, _target


@pytest.mark.v3_toolchain
def test_v3r022_registry_simt_fragment_compiles_for_profile_target(v3_toolchain, tmp_path):
    _, indexed, operation, choice, policy = _linear()
    registry = query_body_tactics(indexed, operation, choice, _target(), numerical_policy=policy)
    tactic = next(item for item in registry.compatible_tactics
                  if item.tactic_id == "simt.k_parallel.w1.v1")
    entry = """
__global__ void registry_probe(Mb3Contraction p) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t output = static_cast<int64_t>(blockIdx.x) + warp;
  simt_output<1>(p, output, lane);
}
"""
    source = tmp_path / "registry_probe.cu"
    output = tmp_path / "registry_probe.o"
    source.write_text(tactic.source_text + entry)
    command = [v3_toolchain, "-std=c++17", "-arch=sm_90a", "-c", str(source), "-o", str(output)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, " ".join(command) + "\n" + result.stdout + result.stderr
    assert output.is_file()
