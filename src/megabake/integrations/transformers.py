"""HuggingFace Transformers integration for megabake."""

import torch
import torch.nn as nn

import megabake
from megabake.schedule_compiler.graph_walker import CompiledModel


class _CausalLMWrapper(nn.Module):
    """Wrap HF CausalLM so forward() returns a plain logits tensor.

    Calls base model + lm_head directly to avoid graph breaks
    in transformers' CausalLM.forward.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.base_model = hf_model.model
        self.lm_head = hf_model.lm_head

    def forward(self, input_ids):
        hidden = self.base_model(input_ids, use_cache=False).last_hidden_state
        return self.lm_head(hidden)


class MegabakeModel:
    """Compiled megakernel for an HF model. Standalone callable."""

    def __init__(
        self,
        compiled: CompiledModel,
        state_dict: dict[str, torch.Tensor],
    ):
        self._compiled = compiled
        self._state_dict = state_dict

    @property
    def compiled(self) -> CompiledModel:
        return self._compiled

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return megabake.run(self._compiled, self._state_dict, input_ids)


def from_pretrained(
    model_name_or_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    sm_version: int | None = None,
    example_seq_len: int = 8,
) -> MegabakeModel:
    """Load HF CausalLM, compile to megakernel, return standalone callable."""
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=dtype,
    ).cuda().eval()

    wrapped = _CausalLMWrapper(hf_model)

    vocab_size = getattr(config, "vocab_size", 32000)
    example_input = torch.randint(
        0, vocab_size, (1, example_seq_len), device="cuda",
    )

    compiled = megabake.compile(wrapped, example_input, sm_version=sm_version, dtype=dtype)

    sd = wrapped.state_dict()
    for name, buf in wrapped.named_buffers():
        if name not in sd:
            sd[name] = buf

    del hf_model

    return MegabakeModel(compiled, sd)
