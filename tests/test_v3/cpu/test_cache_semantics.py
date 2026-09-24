import torch

from megabake.v3.frontend.match_state import CacheSemanticsError, functional_cache_update, validate_kv_cache
from tests.test_v3.fixtures import STATE_POISON


def test_functional_cache_update_preserves_old_storage_and_bounds():
    old = torch.zeros(1, 2, 17, 8)
    update = torch.ones(1, 2, 1, 8)
    new, valid = functional_cache_update(old, update, 16, valid_length=16)
    assert valid == 17 and torch.equal(old, torch.zeros_like(old)) and torch.equal(new[:, :, 16], torch.ones(1, 2, 8))
    with __import__("pytest").raises(CacheSemanticsError):
        functional_cache_update(old, update, 17)


def test_cache_positions_keep_poisoned_untouched_regions_and_validate_gqa_shape():
    for position in (0, 1, 15, 16):
        case = STATE_POISON(position=position)
        before, after = case.state_before["cache_k"], case.state_after["cache_k"]
        assert torch.equal(before[:, :, :position], after[:, :, :position])
        assert torch.equal(before[:, :, position + 1:], after[:, :, position + 1:])
    cache = torch.zeros(1, 2, 17, 8)
    validate_kv_cache(cache, cache.clone(), heads_kv=2, head_dim=8)
    with __import__("pytest").raises(CacheSemanticsError):
        validate_kv_cache(cache, torch.zeros(1, 3, 17, 8), heads_kv=2, head_dim=8)
