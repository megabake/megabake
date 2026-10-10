from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM

from .export import ExportedCapture, capture_exported_program


@dataclass
class TransformerCapture(ExportedCapture):
    model_name: str | None = None
    revision: str | None = None


def capture_transformer(
    model: str | torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    revision: str | None = None,
    device: torch.device | str | None = None,
    backend: Callable[..., Any] | None = None,
    **backend_options: Any,
) -> TransformerCapture:
    model_name = model if isinstance(model, str) else None
    if isinstance(model, str):
        loaded = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            attn_implementation="eager",
        )
    else:
        loaded = model
    target_device = torch.device(device) if device is not None else next(
        loaded.parameters(), torch.empty(0)
    ).device
    loaded = loaded.to(target_device).eval()
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(target_device)
    attention_mask = attention_mask.to(target_device)
    with torch.no_grad():
        exported = torch.export.export(
            loaded,
            (input_ids,),
            kwargs={
                "attention_mask": attention_mask,
                "use_cache": False,
                "return_dict": False,
            },
        )
        capture = capture_exported_program(
            exported,
            args=(input_ids,),
            kwargs={"attention_mask": attention_mask},
            model=loaded,
            backend=backend,
            model_name=model_name or type(loaded).__name__,
            input_defaults={
                "attention_mask": torch.ones_like(input_ids),
                "use_cache": False,
                "return_dict": False,
            },
            **backend_options,
        )
    return TransformerCapture(
        exported_program=capture.exported_program,
        model=capture.model,
        graph=capture.graph,
        context=capture.context,
        forward=capture.forward,
        input_defaults=capture.input_defaults,
        model_name=model_name,
        revision=revision,
    )
