import ctypes
from dataclasses import replace
from types import SimpleNamespace

import pytest

from megabake.v3.backends.cuda.admit import Mb3Contraction, contraction_problem
from megabake.v3.backends.cuda.compile import artifact_key_for
from megabake.v3.backends.cuda.physical import search_physical_candidates
from tests.test_v3.cpu.test_physical_search import _case


def test_v3r024_rejects_wrong_entry_binding_shape_before_cuda_use():
    fake = lambda shape: SimpleNamespace(shape=shape, is_cuda=True)
    with pytest.raises(ValueError, match="LINEAR_TINY shapes"):
        contraction_problem(fake((2, 33)), fake((17, 33)), fake((17,)),
                            fake((2, 17)), alpha=1.75, beta=-0.25)
    assert ctypes.sizeof(Mb3Contraction) == 104


def test_v3r024_artifact_key_includes_numerical_policy_and_shape_guards():
    plan, registry, profile = _case()
    search = search_physical_candidates(plan, registry, profile)
    candidate = next(item for item in search.candidates
                     if item.candidate_id == search.selected_candidate_id)
    tactic = next(item for item in registry.compatible_tactics
                  if item.tactic_id == candidate.tactic_id)
    policy_hash = tactic.numerical_guard["numerical_policy_hash"]
    assert policy_hash
    original = artifact_key_for(candidate, tactic, profile, "source", "nvcc test")
    changed_guard = replace(tactic, numerical_guard={
        **tactic.numerical_guard, "numerical_policy_hash": "different-policy",
    })
    changed = artifact_key_for(candidate, changed_guard, profile, "source", "nvcc test")
    assert changed != original
