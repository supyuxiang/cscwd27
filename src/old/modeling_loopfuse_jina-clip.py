"""LoopFuse on jina-clip-v1.

The text tower is JinaBERT (ALiBi, mean pooling). The vision tower is EVA02
ViT-B/16 (RoPE, CLS pooling). Both towers are embedded once and then the full
encoder is reapplied with shared weights. Pooled states are projected with the
checkpoint's own projections and fused across loops.

Each later loop keeps that first embedding as its base. A scalar gate,
initialized at zero, mixes in the normalized previous hidden state. With the
gate closed, every loop sees the same input as the first pass. The fusion
head adds a learned loop position, keeps the first-loop feature for the
classifier, and starts as an identity around attention and the feed-forward.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
import transformers.models.clip.modeling_clip as clip_modeling


def _install_clip_loss() -> None:
    """transformers 5 removed clip_loss; jina-clip-v1 still imports it."""
    if hasattr(clip_modeling, "clip_loss"):
        return

    def clip_loss(logits: torch.Tensor) -> torch.Tensor:
        target = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2

    clip_modeling.clip_loss = clip_loss


_install_clip_loss()

MAX_LOOPS = 16


def _gated_reentry(original: torch.Tensor, previous: torch.Tensor, norm: nn.LayerNorm, gate: torch.Tensor) -> torch.Tensor:
    """original + gate * (norm(previous) - original). Gate 0 leaves original unchanged."""
    return original + gate * (norm(previous) - original)


def _embedding_width(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Embedding):
            return int(child.embedding_dim)
    for child in module.modules():
        if isinstance(child, nn.LayerNorm):
            return int(child.normalized_shape[0])
    raise RuntimeError("Cannot infer the hidden size used for loop reentry.")


class GQA(nn.Module):
    def __init__(self, d_model: int, num_heads_q: int, num_heads_kv: int):
        super().__init__()
        if num_heads_q % num_heads_kv != 0:
            raise ValueError(f"num_heads_q must be divisible by num_heads_kv, got {num_heads_q}, {num_heads_kv}")
        if d_model % num_heads_q != 0:
            raise ValueError(f"d_model must be divisible by num_heads_q, got {d_model}, {num_heads_q}")
        self.d_model = d_model
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.d_k = d_model // num_heads_q
        self.wq = nn.Linear(d_model, self.d_k * num_heads_q, bias=False)
        self.wk = nn.Linear(d_model, self.d_k * num_heads_kv, bias=False)
        self.wv = nn.Linear(d_model, self.d_k * num_heads_kv, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.wo.weight)
        nn.init.zeros_(self.wo.bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        bsz, seq_len_q, seq_len_kv = q.size(0), q.size(1), k.size(1)
        q = self.wq(q).view(bsz, seq_len_q, self.num_heads_q, self.d_k).transpose(1, 2)
        k = self.wk(k).view(bsz, seq_len_kv, self.num_heads_kv, self.d_k).transpose(1, 2)
        v = self.wv(v).view(bsz, seq_len_kv, self.num_heads_kv, self.d_k).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, enable_gqa=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len_q, -1)
        return self.wo(attn_out)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_ffn, d_model)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


def _stack_loop_features(outputs) -> torch.Tensor:
    if isinstance(outputs, BaseModelOutputWithPooling):
        features = outputs.pooler_output
    elif torch.is_tensor(outputs):
        features = outputs
    else:
        pooled = []
        for item in outputs:
            if isinstance(item, BaseModelOutputWithPooling):
                pooled.append(item.pooler_output)
            elif torch.is_tensor(item):
                pooled.append(item)
            else:
                raise TypeError(f"Unsupported loop feature type: {type(item).__name__}")
        if not pooled:
            raise ValueError("Loop feature list is empty.")
        return torch.stack(pooled, dim=1)
    if features.dim() == 2:
        return features.unsqueeze(1)
    if features.dim() != 3:
        raise ValueError(
            f"Pooled loop features must be [batch, dim] or [batch, loops, dim], got {tuple(features.shape)}"
        )
    return features


class Fuse(nn.Module):
    def __init__(self, embed_dim: int, num_heads_q: int, num_heads_kv: int, dropout_p: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.loop_pos = nn.Parameter(torch.zeros(MAX_LOOPS, embed_dim))
        nn.init.normal_(self.loop_pos, std=0.02)
        self.gqa_t_v = GQA(embed_dim, num_heads_q, num_heads_kv)
        self.gqa_v_t = GQA(embed_dim, num_heads_q, num_heads_kv)
        self.ffn_t_v = FFN(embed_dim, 4 * embed_dim, dropout_p)
        self.ffn_v_t = FFN(embed_dim, 4 * embed_dim, dropout_p)
        self.ln_t_v = nn.LayerNorm(embed_dim)
        self.ln_v_t = nn.LayerNorm(embed_dim)

    def _mix(self, query: torch.Tensor, key: torch.Tensor, gqa: GQA, ffn: FFN, norm: nn.LayerNorm) -> torch.Tensor:
        attended = query + gqa(query, key, key)
        return attended + ffn(norm(attended))

    def forward(self, out_ls_t, out_ls_v) -> tuple[torch.Tensor, torch.Tensor]:
        text = _stack_loop_features(out_ls_t)
        vision = _stack_loop_features(out_ls_v)
        dtype = self.ln_t_v.weight.dtype
        text = text.to(dtype=dtype)
        vision = vision.to(dtype=dtype)
        if text.size(-1) != self.embed_dim or vision.size(-1) != self.embed_dim:
            raise ValueError(
                f"Fuse expected embed_dim={self.embed_dim}, got text {tuple(text.shape)} and vision {tuple(vision.shape)}."
            )
        n_loops = text.size(1)
        if n_loops > MAX_LOOPS or vision.size(1) != n_loops:
            raise ValueError(f"Fuse supports 1..{MAX_LOOPS} loops, got text {text.size(1)} and vision {vision.size(1)}.")
        text_first = text[:, 0]
        vision_first = vision[:, 0]
        loop_pos = self.loop_pos[:n_loops].to(dtype=dtype).unsqueeze(0)
        text = text + loop_pos
        vision = vision + loop_pos
        text_out = self._mix(text, vision, self.gqa_t_v, self.ffn_t_v, self.ln_t_v)
        vision_out = self._mix(vision, text, self.gqa_v_t, self.ffn_v_t, self.ln_v_t)
        return torch.cat([text_first, text_out[:, -1]], dim=-1), torch.cat([vision_first, vision_out[:, -1]], dim=-1)


class LoopFuse(nn.Module):
    def __init__(
        self,
        clip_path: str,
        num_heads_q: int,
        num_heads_kv: int,
        dropout_p: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self._load_looped_models(clip_path)
        self.fuse = Fuse(self.projection_dim, num_heads_q, num_heads_kv, dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(self.projection_dim * 4, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.projection_dim, num_classes),
        )

    def _load_looped_models(self, clip_path: str) -> None:
        official = AutoModel.from_pretrained(clip_path, trust_remote_code=True)
        text_model = official.text_model
        vision_model = official.vision_model
        text_projection = official.text_projection
        visual_projection = official.visual_projection
        del official.text_model, official.vision_model, official.text_projection, official.visual_projection
        self.text_model = text_model
        self.vision_model = vision_model
        self.text_projection = text_projection
        self.visual_projection = visual_projection
        self.projection_dim = int(official.projection_dim)
        encoder = getattr(self.text_model.transformer, "encoder", None)
        if encoder is not None and getattr(encoder, "use_flash_attn", False):
            encoder.use_flash_attn = False
        rope = getattr(self.vision_model, "rope", None)
        if rope is not None:
            self.vision_model._loopfuse_rope_forward = rope.forward
        text_width = _embedding_width(self.text_model.transformer.embeddings)
        vision_width = int(self.vision_model.cls_token.shape[-1])
        self.text_loop_gate = nn.Parameter(torch.zeros(1))
        self.text_reentry_norm = nn.LayerNorm(text_width)
        self.vision_loop_gate = nn.Parameter(torch.zeros(1))
        self.vision_reentry_norm = nn.LayerNorm(vision_width)
        self.processor = AutoProcessor.from_pretrained(clip_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)
        self._contiguous_parameters()

    def _contiguous_parameters(self) -> None:
        for module in self.modules():
            for name, param in list(module.named_parameters(recurse=False)):
                if param.is_contiguous():
                    continue
                replacement = nn.Parameter(param.detach().contiguous(), requires_grad=param.requires_grad)
                setattr(module, name, replacement)

    def _loop_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_loops: int) -> list[torch.Tensor]:
        transformer = self.text_model.transformer
        original = transformer.embeddings(input_ids)
        if not transformer.fused_dropout_add_ln:
            original = transformer.emb_ln(original)
        original = transformer.emb_drop(original)
        keep = attention_mask
        if keep.dtype != torch.long:
            keep = keep.long()
        hidden_states = original
        outputs = []
        for step in range(num_loops):
            sequence = torch.utils.checkpoint.checkpoint(
                lambda hidden, mask=keep: transformer.encoder(hidden, key_padding_mask=mask),
                hidden_states,
                use_reentrant=False,
            )
            pooled = self.text_model.proj(self.text_model.pooler(BaseModelOutput(last_hidden_state=sequence), keep))
            hidden_states = _gated_reentry(original, sequence, self.text_reentry_norm, self.text_loop_gate)
            # The reentry is only consumed by the next loop. On the last step the
            # added term is exactly zero, so the pooled value is unchanged, but the
            # gate and reentry norm stay in the loss graph. One-loop DDP otherwise
            # reports them as unused parameters.
            if step == num_loops - 1:
                pooled = pooled + (hidden_states - original).sum() * 0
            outputs.append(self.text_projection(pooled))
        return outputs

    def _loop_vision(self, pixel_values: torch.Tensor, num_loops: int) -> list[torch.Tensor]:
        vision = self.vision_model
        hidden_states = vision.patch_embed(pixel_values)
        batch_size = hidden_states.size(0)
        cls_tokens = vision.cls_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
        if vision.pos_embed is not None:
            hidden_states = hidden_states + vision.pos_embed
        hidden_states = vision.pos_drop(hidden_states)
        if vision.rope is not None:
            if vision.training and not isinstance(vision.patch_dropout, nn.Identity):
                hidden_states, patch_indices_keep = vision.patch_dropout(hidden_states)
            else:
                patch_indices_keep = None
                hidden_states = vision.patch_dropout(hidden_states)
            vision.rope.forward = partial(
                vision._loopfuse_rope_forward, patch_indices_keep=patch_indices_keep
            )
        else:
            hidden_states = vision.patch_dropout(hidden_states)
        rel_pos_bias = vision.rel_pos_bias() if vision.rel_pos_bias is not None else None
        original = hidden_states
        outputs = []
        blocks = vision.blocks
        bias = rel_pos_bias

        def _run_blocks(tokens):
            for block in blocks:
                tokens = block(tokens, rel_pos_bias=bias)
            return tokens

        for step in range(num_loops):
            tokens = torch.utils.checkpoint.checkpoint(_run_blocks, hidden_states, use_reentrant=False)
            pooled_tokens = vision.norm(tokens)
            if vision.fc_norm is not None:
                pooled = vision.fc_norm(pooled_tokens.mean(dim=1))
            else:
                pooled = pooled_tokens[:, 0]
            hidden_states = _gated_reentry(original, tokens, self.vision_reentry_norm, self.vision_loop_gate)
            # The reentry is only consumed by the next loop. On the last step the
            # added term is exactly zero, so the pooled value is unchanged, but the
            # gate and reentry norm stay in the loss graph. One-loop DDP otherwise
            # reports them as unused parameters.
            if step == num_loops - 1:
                pooled = pooled + (hidden_states - original).sum() * 0
            outputs.append(self.visual_projection(vision.head(pooled)))
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        num_loops: int = 3,
    ) -> torch.Tensor:
        if num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")
        text_loops = self._loop_text(input_ids, attention_mask, num_loops)
        vision_loops = self._loop_vision(pixel_values, num_loops)
        feature_t_v, feature_v_t = self.fuse(text_loops, vision_loops)
        return self.classifier(torch.cat([feature_t_v, feature_v_t], dim=-1))


__all__ = ["LoopFuse", "Fuse"]
