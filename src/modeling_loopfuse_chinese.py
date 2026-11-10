"""LoopFuse on Chinese CLIP.

The text tower is Chinese CLIP's BERT encoder. The vision tower is its ViT.
Both are looped with shared weights, then projected into the Chinese CLIP
embedding space and fused across loops.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, ChineseCLIPModel, ChineseCLIPTextModel, ChineseCLIPVisionModel
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


class LoopedChineseCLIPTextModel(ChineseCLIPTextModel):
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_loops: int = 3,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[BaseModelOutputWithPooling]:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        if num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        outputs = []
        for _ in range(num_loops):
            encoder_outputs: BaseModelOutput = self.encoder(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
            last_hidden_state = encoder_outputs.last_hidden_state
            outputs.append(
                BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=last_hidden_state[:, 0],
                )
            )
            hidden_states = last_hidden_state
        return outputs


class LoopedChineseCLIPVisionModel(ChineseCLIPVisionModel):
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        num_loops: int = 3,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[BaseModelOutputWithPooling]:
        if num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)
        outputs = []
        for _ in range(num_loops):
            encoder_outputs: BaseModelOutput = self.encoder(inputs_embeds=hidden_states, **kwargs)
            last_hidden_state = encoder_outputs.last_hidden_state
            pooled_output = self.post_layernorm(last_hidden_state[:, 0, :])
            outputs.append(
                BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooled_output,
                )
            )
            hidden_states = last_hidden_state
        return outputs


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
        use_flash = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
        flash_ctx = (
            torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION])
            if use_flash
            else contextlib.nullcontext()
        )
        with flash_ctx:
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
        raise ValueError(f"Pooled loop features must be [batch, dim] or [batch, loops, dim], got {tuple(features.shape)}")
    return features


def _encoder_weights(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    head = prefix + "."
    weights = {}
    for key, value in state_dict.items():
        if not key.startswith(head):
            continue
        subkey = key[len(head):]
        if subkey.endswith("position_ids") or subkey.endswith("token_type_ids"):
            continue
        weights[subkey] = value
    if not weights:
        raise RuntimeError(f"Checkpoint has no weights under {prefix!r}.")
    return weights


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
        official = ChineseCLIPModel.from_pretrained(clip_path, trust_remote_code=True)
        state_dict = official.state_dict()
        config = official.config
        del official

        self.text_encoder = LoopedChineseCLIPTextModel(config.text_config)
        self.vision_encoder = LoopedChineseCLIPVisionModel(config.vision_config)
        self.text_encoder.load_state_dict(_encoder_weights(state_dict, "text_model"), strict=True)
        self.vision_encoder.load_state_dict(_encoder_weights(state_dict, "vision_model"), strict=True)

        self.text_hidden = int(config.text_config.hidden_size)
        self.vision_hidden = int(config.vision_config.hidden_size)
        self.projection_dim = int(config.projection_dim)
        self.text_projection = nn.Linear(self.text_hidden, self.projection_dim, bias=False)
        self.visual_projection = nn.Linear(self.vision_hidden, self.projection_dim, bias=False)
        if "text_projection.weight" not in state_dict or "visual_projection.weight" not in state_dict:
            raise RuntimeError("Checkpoint is missing Chinese CLIP text_projection or visual_projection.")
        with torch.no_grad():
            self.text_projection.weight.copy_(state_dict["text_projection.weight"])
            self.visual_projection.weight.copy_(state_dict["visual_projection.weight"])

        self.processor = AutoProcessor.from_pretrained(clip_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        num_loops: int = 3,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_outs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            num_loops=num_loops,
        )
        vision_outs = self.vision_encoder(pixel_values=pixel_values, num_loops=num_loops)
        text_loops = [self.text_projection(out.pooler_output) for out in text_outs]
        vision_loops = [self.visual_projection(out.pooler_output) for out in vision_outs]
        feature_t_v, feature_v_t = self.fuse(text_loops, vision_loops)
        return self.classifier(torch.cat([feature_t_v, feature_v_t], dim=-1))


__all__ = [
    "LoopedChineseCLIPTextModel",
    "LoopedChineseCLIPVisionModel",
    "Fuse",
    "LoopFuse",
]
