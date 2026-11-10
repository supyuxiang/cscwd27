"""LoopFuse on jina-clip-v1.

The text tower is JinaBERT (ALiBi, mean pooling). The vision tower is EVA02
ViT-B/16 (RoPE, CLS pooling). Both towers are embedded once and then the full
encoder is reapplied with shared weights. Pooled states are projected with the
checkpoint's own projections and fused across loops.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling


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
        self.gqa_t_v = GQA(embed_dim, num_heads_q, num_heads_kv)
        self.gqa_v_t = GQA(embed_dim, num_heads_q, num_heads_kv)
        self.ffn_t_v = FFN(embed_dim, 4 * embed_dim, dropout_p)
        self.ffn_v_t = FFN(embed_dim, 4 * embed_dim, dropout_p)
        self.ln_t_v = nn.LayerNorm(embed_dim)
        self.ln_v_t = nn.LayerNorm(embed_dim)

    def forward(self, out_ls_t, out_ls_v) -> tuple[torch.Tensor, torch.Tensor]:
        text = _stack_loop_features(out_ls_t)
        vision = _stack_loop_features(out_ls_v)
        if text.size(-1) != self.embed_dim or vision.size(-1) != self.embed_dim:
            raise ValueError(
                f"Fuse expected embed_dim={self.embed_dim}, got text {tuple(text.shape)} and vision {tuple(vision.shape)}."
            )
        text_out = self.ffn_t_v(self.ln_t_v(self.gqa_t_v(text, vision, vision)))
        vision_out = self.ffn_v_t(self.ln_v_t(self.gqa_v_t(vision, text, text)))
        return text_out[:, -1], vision_out[:, -1]


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
            nn.Linear(self.projection_dim * 2, self.projection_dim),
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
        self.processor = AutoProcessor.from_pretrained(clip_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

    def _loop_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_loops: int) -> list[torch.Tensor]:
        transformer = self.text_model.transformer
        hidden_states = transformer.embeddings(input_ids)
        if not transformer.fused_dropout_add_ln:
            hidden_states = transformer.emb_ln(hidden_states)
        hidden_states = transformer.emb_drop(hidden_states)
        keep = attention_mask
        if keep.dtype != torch.long:
            keep = keep.long()
        outputs = []
        for _ in range(num_loops):
            sequence = transformer.encoder(hidden_states, key_padding_mask=keep)
            pooled = self.text_model.pooler(BaseModelOutput(last_hidden_state=sequence), keep)
            outputs.append(self.text_projection(self.text_model.proj(pooled)))
            hidden_states = sequence
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
        outputs = []
        for _ in range(num_loops):
            tokens = hidden_states
            for block in vision.blocks:
                tokens = block(tokens, rel_pos_bias=rel_pos_bias)
            pooled_tokens = vision.norm(tokens)
            if vision.fc_norm is not None:
                pooled = vision.fc_norm(pooled_tokens.mean(dim=1))
            else:
                pooled = pooled_tokens[:, 0]
            outputs.append(self.visual_projection(vision.head(pooled)))
            hidden_states = tokens
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
