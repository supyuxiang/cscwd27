"""Frozen CLIP token encoders in the shared projection space."""

from __future__ import annotations

import torch
from torch import nn


def _load_clip(name: str):
    lowered = name.lower()

    if "chinese" in lowered:
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        model = ChineseCLIPModel.from_pretrained(name)
        processor = ChineseCLIPProcessor.from_pretrained(name)
        return model, processor
    
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(name)
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor


def _projection_dim(clip: nn.Module) -> int:
    dims = []
    for name in ("text_projection", "visual_projection"):
        projection = getattr(clip, name, None)
        if isinstance(projection, nn.Linear):
            dims.append(int(projection.out_features))
        elif torch.is_tensor(projection):
            dims.append(int(projection.shape[-1]))
    if not dims:
        return int(clip.config.projection_dim)
    if len(set(dims)) != 1:
        raise ValueError(f"Text/image projection dims differ: {dims}")
    return dims[0]


def _apply_projection(projection: nn.Module | torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor:
    if projection is None:
        return hidden
    if isinstance(projection, nn.Linear):
        return projection(hidden)
    if not torch.is_tensor(projection):
        raise TypeError(f"Unsupported projection type: {type(projection)!r}")
    return hidden @ projection.to(dtype=hidden.dtype)


def _vision_patches(vision_module: nn.Module, outputs) -> torch.Tensor:
    hidden = outputs.last_hidden_state
    layer_norm = getattr(vision_module, "post_layernorm", None)
    if layer_norm is None:
        inner = getattr(vision_module, "vision_model", None)
        layer_norm = getattr(inner, "post_layernorm", None) if inner is not None else None
    if layer_norm is not None:
        hidden = layer_norm(hidden)
    if hidden.size(1) > 1:
        hidden = hidden[:, 1:, :]
    return hidden


class CLIPEncoder(nn.Module):
    def __init__(self, clip_name: str, d_model: int, freeze: bool = True) -> None:
        super().__init__()
        clip, processor = _load_clip(clip_name)
        self.clip = clip
        self.processor = processor
        self.freeze = freeze
        if freeze:
            for parameter in self.clip.parameters():
                parameter.requires_grad = False
            self.clip.eval()
        projection_dim = _projection_dim(self.clip)
        self.text_align = nn.Linear(projection_dim, d_model) if projection_dim != d_model else None
        self.image_align = nn.Linear(projection_dim, d_model) if projection_dim != d_model else None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.clip.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(not self.freeze):
            text_tokens, image_tokens = self._tokens(input_ids, attention_mask, pixel_values.float())
        if self.freeze:
            text_tokens = text_tokens.detach()
            image_tokens = image_tokens.detach()
        if self.text_align is not None:
            text_tokens = self.text_align(text_tokens)
        if self.image_align is not None:
            image_tokens = self.image_align(image_tokens)
        text_pad = attention_mask.eq(0)
        if text_pad.size(1) != text_tokens.size(1):
            raise ValueError(
                f"Text mask length {text_pad.size(1)} does not match tokens {text_tokens.size(1)}"
            )
        return text_tokens, image_tokens, text_pad

    def _tokens(self, input_ids, attention_mask, pixel_values):
        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        text_tokens = _apply_projection(self.clip.text_projection, text_out.last_hidden_state)
        image_tokens = _apply_projection(
            self.clip.visual_projection,
            _vision_patches(self.clip.vision_model, vision_out),
        )
        return text_tokens, image_tokens
