import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from megabake.bodies import SelectionPolicy, catalog
from megabake.compiler import _search_scheduler
from megabake.plan import PlanSearchResult, SearchBudget
from megabake.planner import _contraction_body_order


def test_table_gemm_contracts_are_registered_with_their_generations():
    blackwell = catalog.table_gemm("sm100_tma")
    assert catalog.is_table_gemm("g0_range")
    assert catalog.table_gemm("g0_range") is None
    assert blackwell is not None
    assert blackwell.requires_table
    assert not blackwell.supports_streaming
    assert blackwell.supports_mixed_tiles
    assert blackwell.full_operand_descriptors
    assert blackwell.direct_store_with_splits
    assert not catalog.is_table_gemm("generic")
    assert catalog.table_gemm("generic") is None


def test_table_gemm_registry_accepts_a_new_generation_body(monkeypatch):
    extra = SimpleNamespace(provider_name="new_gemm")
    first = catalog.GENERATIONS[0]
    monkeypatch.setattr(
        catalog,
        "GENERATIONS",
        (replace(first, table_gemms=(*first.table_gemms, extra)), *catalog.GENERATIONS[1:]),
    )
    assert catalog.table_gemm("new_gemm") is extra


def test_planner_accepts_provider_selection_priority():
    body = SimpleNamespace(
        provider="new_provider",
        selection_priority=-3,
        participants=SimpleNamespace(participating_threads=128, registers_per_thread=32),
        scratch=SimpleNamespace(size_bytes=0),
        evidence=SimpleNamespace(measurements={}),
    )
    assert _contraction_body_order(body, None, SelectionPolicy())[0] == -3


def test_core_dispatch_does_not_hardcode_table_gemm_providers():
    body_names = {
        body.provider_name for generation in catalog.GENERATIONS for body in generation.table_gemms
    }
    core = Path(__file__).resolve().parents[2] / "src" / "megabake"
    provider_modules = {
        type(body).__module__.rsplit(".", 1)[0]
        for generation in catalog.GENERATIONS
        for body in generation.table_gemms
    }
    hardware_modules = tuple(
        provider_modules | {name.removeprefix("megabake.") for name in provider_modules}
    )
    for name in ("codegen.py", "compiler.py", "plan.py", "planner.py"):
        tree = ast.parse((core / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in body_names, (name, node.lineno, node.value)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(hardware_modules)
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(hardware_modules) for alias in node.names)


@pytest.mark.parametrize(
    ("major", "minor", "architecture", "wgmma"),
    [(9, 0, "sm90a", True), (10, 0, "sm100", False), (12, 1, "sm121", False)],
)
def test_cuda_target_facts_come_from_generation(major, minor, architecture, wgmma):
    properties = SimpleNamespace(
        major=major,
        minor=minor,
        warp_size=32,
        max_threads_per_block=1024,
        shared_memory_per_block_optin=200_000,
        multi_processor_count=100,
    )
    target = catalog.cuda_target(properties)
    assert target.architecture == architecture
    assert target.supports_wgmma is wgmma
    assert catalog.supported_architecture(architecture)


def test_cuda_target_rejects_unregistered_device():
    with pytest.raises(ValueError, match="unsupported CUDA architecture"):
        catalog.cuda_target(SimpleNamespace(major=8, minor=0))


def test_event_scheduler_preserves_launch_options(monkeypatch):
    options = object()
    seen = []

    def search(graph, target, policy, budget):
        seen.append(policy)
        return PlanSearchResult(None, None, (), (), False, None)

    monkeypatch.setattr("megabake.compiler.search_plans", search)
    monkeypatch.setattr("megabake.compiler._make_plan_evaluator", lambda *args: object())
    policy = SelectionPolicy(table_launch=True, target_options=options)
    _search_scheduler(None, None, policy, SearchBudget(), None, None, None, "region-events")
    assert seen[0].event_schedule
    assert seen[0].table_launch
    assert seen[0].target_options is options
