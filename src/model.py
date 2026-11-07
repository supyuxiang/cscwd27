"""CMAAN: CLIP features, MS-Fuse, dual-stage attention, and a veracity head."""

from __future__ import annotations

import torch
from torch import nn

from src.config import CSCWDConfig
from src.utils import masked_mean
from src.workers import BidirectionalAttention, CLIPEncoder, CrossAttentionEncoder, Detector, MSFuse
from src.workers.fusion import concatenate_modalities


class CMAAN(nn.Module):
    def __init__(self, config: CSCWDConfig) -> None:
        super().__init__()
        self.config = config
        dim = int(config.d_model)
        self.use_fusion = bool(config.use_fusion)
        self.use_stage1 = bool(config.use_stage1)
        self.use_stage2 = bool(config.use_stage2)

        self.encoder = CLIPEncoder(
            clip_name=config.clip_name,
            d_model=dim,
            freeze=bool(config.freeze_clip)
        )
        self.processor = self.encoder.processor
        self.fusion = MSFuse()
        block = dict(
            dim=dim,
            n_heads=int(config.n_heads),
            n_layers=int(config.n_layers),
            dropout=float(config.dropout),
            ffn_ratio=int(config.ffn_ratio),
        )
        self.stage1_text = CrossAttentionEncoder(**block)
        self.stage1_image = CrossAttentionEncoder(**block)
        self.stage2 = BidirectionalAttention(dim)

        n_vectors = 2 * ((1 if self.use_stage1 else 0) + (1 if self.use_stage2 else 0))
        if n_vectors == 0:
            n_vectors = 2

        self.detector = Detector(
            in_dim=dim * n_vectors,
            hidden=dim,
            num_classes=config.num_classes,
            dropout=float(config.dropout)
        )
        
    @classmethod
    def from_config(cls, config: CSCWDConfig) -> "CMAAN":
        return cls(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        text, image, text_pad = self.encoder(input_ids, attention_mask, pixel_values)
        if self.use_fusion:
            fused, fused_pad = self.fusion(text, image, text_pad)
        else:
            fused, fused_pad = concatenate_modalities(text, image, text_pad)

        text_stage1 = text
        image_stage1 = image
        if self.use_stage1:
            text_stage1 = self.stage1_text(text, fused, text_pad, fused_pad)
            image_stage1 = self.stage1_image(image, fused, None, fused_pad)

        text_stage2 = text_stage1
        image_stage2 = image_stage1
        if self.use_stage2:
            text_stage2, image_stage2 = self.stage2(text_stage1, image_stage1, text_pad)

        parts: list[torch.Tensor] = []
        if self.use_stage1:
            parts.extend(
                [
                    masked_mean(text_stage1, text_pad),
                    masked_mean(image_stage1, None),
                ]
            )
        if self.use_stage2:
            parts.extend(
                [
                    masked_mean(text_stage2, text_pad),
                    masked_mean(image_stage2, None),
                ]
            )
        if not parts:
            parts.extend([masked_mean(text, text_pad), masked_mean(image, None)])
        return self.detector(torch.cat(parts, dim=-1))
