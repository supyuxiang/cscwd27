"""Dual-stage cross-modal attention.

Stage 1 is a post-norm transformer stack. Unimodal tokens are queries. The
fused sequence supplies keys and values. Figure 2 puts a feed-forward network
after the attention, with a residual and layer norm on both sublayers.

Stage 2 is the shared bilinear map from the paper, not a second pair of
multi-head attentions:

    S = T' W_b (V')^T
    A^{v->t} = softmax(S) over image positions
    T_hat = A^{v->t} V'
    A^{t->v} = softmax(S^T) over text positions
    V_hat = A^{t->v} T'
"""

from __future__ import annotations

import torch
from torch import nn



class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float, ffn_ratio: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"d_model ({dim}) must be divisible by n_heads ({n_heads})")
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        hidden = dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pad: torch.Tensor | None = None,
        memory_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        updated, _ = self.attn(
            query,
            memory,
            memory,
            key_padding_mask=_safe_key_padding(memory_pad),
            need_weights=False,
        )
        hidden = self.norm1(query + self.dropout(updated))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        if query_pad is not None:
            hidden = hidden.masked_fill(query_pad.unsqueeze(-1), 0.0)
        return hidden


class CrossAttentionEncoder(nn.Module):
    """Stack of cross-attention blocks. Memory stays the fused sequence."""

    def __init__(self, dim: int, n_heads: int, n_layers: int, dropout: float, ffn_ratio: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(dim, n_heads, dropout, ffn_ratio) for _ in range(n_layers)]
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pad: torch.Tensor | None = None,
        memory_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = query
        for layer in self.layers:
            hidden = layer(hidden, memory, query_pad, memory_pad)
        return hidden


class BidirectionalAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.weight)

    def similarity(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        projected = torch.matmul(text, self.weight)
        return torch.matmul(projected, image.transpose(-1, -2))

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.similarity(text, image)
        image_to_text = _masked_softmax(scores, None)
        text_hat = torch.matmul(image_to_text, image)

        text_key_pad = None if text_pad is None else text_pad.unsqueeze(1)
        text_to_image = _masked_softmax(scores.transpose(-1, -2), text_key_pad)
        image_hat = torch.matmul(text_to_image, text)
        if text_pad is not None:
            text_hat = text_hat.masked_fill(text_pad.unsqueeze(-1), 0.0)
        return text_hat, image_hat


def _masked_softmax(logits: torch.Tensor, key_pad: torch.Tensor | None) -> torch.Tensor:
    if key_pad is not None:
        logits = logits.masked_fill(key_pad, torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=-1)
    if key_pad is None:
        return weights
    weights = weights.masked_fill(key_pad, 0.0)
    all_masked = key_pad.all(dim=-1, keepdim=True)
    return weights.masked_fill(all_masked, 0.0)


def _safe_key_padding(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None or mask.numel() == 0:
        return mask
    all_pad = mask.all(dim=1)
    if not bool(all_pad.any()):
        return mask
    safe = mask.clone()
    safe[all_pad, 0] = False
    return safe
