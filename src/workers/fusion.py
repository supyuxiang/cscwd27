"""Parameter-free fusion of two CLIP token sequences.

Equation (1) pools each modality, concatenates the pooled vectors, and applies
softmax. A flat softmax over the 2d concatenated vector cannot be multiplied
with a d-dimensional feature, and it would shrink every channel by about
1/(2d). Here the pooled vectors are stacked and softmax is taken over the
modality axis, so each channel gets two gates that sum to 1. Those gates are
the two slices of the one softmax that equation (1) writes twice. Text and
image sequences have different lengths, so the fused memory in Figure 1 is the
gated tokens concatenated on the sequence axis. A single summed vector would
make the stage-1 softmax identically one.
"""

from __future__ import annotations

import torch
from torch import nn

from src.utils import masked_mean


class MSFuse(nn.Module):
    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text.size(-1) != image.size(-1):
            raise ValueError(f"Feature dims differ: text {text.size(-1)} image {image.size(-1)}")
        pooled_text = masked_mean(text, text_pad)
        pooled_image = masked_mean(image, None)
        weights = torch.softmax(torch.stack([pooled_text, pooled_image], dim=1), dim=1)
        text_gate = weights[:, 0]
        image_gate = weights[:, 1]
        gated_text = text * text_gate.unsqueeze(1)
        gated_image = image * image_gate.unsqueeze(1)
        if text_pad is not None:
            gated_text = gated_text.masked_fill(text_pad.unsqueeze(-1), 0.0)
        return concatenate_modalities(gated_text, gated_image, text_pad)


def concatenate_modalities(
    text: torch.Tensor,
    image: torch.Tensor,
    text_pad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_pad = torch.zeros(
        image.size(0),
        image.size(1),
        dtype=torch.bool,
        device=image.device,
    )
    if text_pad is None:
        text_pad = torch.zeros(
            text.size(0),
            text.size(1),
            dtype=torch.bool,
            device=text.device,
        )
    return torch.cat([text, image], dim=1), torch.cat([text_pad, image_pad], dim=1)
