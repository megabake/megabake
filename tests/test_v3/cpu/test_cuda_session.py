from types import SimpleNamespace

import pytest
import torch

from megabake.v3.contracts import StepABI
from megabake.v3.backends.cuda.session import (
    CachedStepSessionError,
    _has_owned_storage_root,
    _storage_key,
    validate_cached_step_inputs,
)


def _abi(*, capacity=4):
    return SimpleNamespace(
        ordered_user_inputs=({"placeholder": "token"}, {"placeholder": "cache"}),
        guard_set={
            "shapes": {"token": [1, 1], "cache": [1, 2, 4, 3]},
            "strides": {"token": [1, 1], "cache": [24, 12, 3, 1]},
            "dtypes": {"token": "int64", "cache": "float16", "internal": "float32"},
        },
        old_state_inputs=({
            "placeholder": "cache",
            "layout": "layers,kv,capacity,head_dim",
            "capacity": capacity,
        },),
    )


def _inputs():
    return {
        "token": torch.zeros((1, 1), dtype=torch.int64),
        "cache": torch.zeros((1, 2, 4, 3), dtype=torch.float16),
    }


def test_cached_step_input_guards_accept_exact_tensors():
    validate_cached_step_inputs(_abi(), _inputs())


@pytest.mark.parametrize("change, message", [
    (lambda values: values.update(token=torch.zeros((1, 2), dtype=torch.int64)), "shape guard"),
    (lambda values: values.update(cache=torch.zeros((1, 2, 4, 3), dtype=torch.float32)), "dtype guard"),
    (lambda values: values.update(cache=torch.empty_strided(
        (1, 2, 4, 3), (24, 1, 3, 8), dtype=torch.float16)), "stride guard"),
    (lambda values: values.update(extra=torch.zeros(1)), "exactly"),
])
def test_cached_step_input_guards_reject_metadata_or_signature_mismatch(change, message):
    values = _inputs()
    change(values)
    with pytest.raises(CachedStepSessionError, match=message):
        validate_cached_step_inputs(_abi(), values)


def test_cached_step_capacity_guard_rejects_inconsistent_state_contract():
    with pytest.raises(CachedStepSessionError, match="capacity guard"):
        validate_cached_step_inputs(_abi(capacity=5), _inputs())


def test_cached_step_capacity_guard_rejects_layout_rank_mismatch():
    abi = _abi()
    abi.old_state_inputs = ({
        "placeholder": "cache",
        "layout": "layers,kv,head_dim,aux,capacity",
        "capacity": 4,
    },)
    with pytest.raises(CachedStepSessionError, match="layout rank"):
        validate_cached_step_inputs(abi, _inputs())


def test_cached_step_owned_output_accepts_view_of_fresh_allocation():
    values = {
        "root": SimpleNamespace(alias_kind="fresh", alias_set="storage:root", alias_sources=()),
        "view": SimpleNamespace(alias_kind="view", alias_set="storage:root", alias_sources=("root",)),
    }

    assert _has_owned_storage_root("view", values)


@pytest.mark.parametrize("kind,sources,alias_set", [
    ("state_input", (), "state:kv"),
    ("unknown", (), None),
    ("view", ("missing",), "state:kv"),
])
def test_cached_step_owned_output_rejects_unproved_root(kind, sources, alias_set):
    values = {"output": SimpleNamespace(alias_kind=kind, alias_set=alias_set, alias_sources=sources)}

    assert not _has_owned_storage_root("output", values)


def test_cached_step_output_storage_check_covers_offset_views():
    source = torch.arange(8)

    assert _storage_key(source) == _storage_key(source[2:])
    assert _storage_key(source) != _storage_key(torch.arange(8))


def test_indexed_program_hash_covers_step_abi_guards():
    from tests.test_v3.worker_fixtures import capture_unfamiliar_block

    captured, indexed, _, _, _ = capture_unfamiliar_block()
    original_hash = indexed.structural_hash
    contract = captured.step_abi.to_dict()
    contract["guard_set"]["shapes"]["x"] = [1, 5]
    captured.step_abi = StepABI.from_dict(contract)

    assert indexed.structural_hash != original_hash
