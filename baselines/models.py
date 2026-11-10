"""Same-backbone baselines for the current whole-tower LoopFuse.

LoopFuse repeats the full text tower and the full vision tower. These
baselines use that same checkpoint, the same projections, and the same
fusion head, so a difference in accuracy is attributable to the encoding
pattern rather than to a different backbone.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask

from .lora import LoopLoRABank, attention_linears


def resolve_backbone(config: dict, model_type: str) -> str:
    requested = str(config.get("backbone", "")).lower()
    if requested in {"clip", "chinese", "jina"}:
        return requested
    kind = model_type.lower()
    clip_name = str(config.get("clip_name", "")).lower()
    if kind in {"jina", "jina-clip", "jinaclip"} or "jina-clip" in clip_name:
        return "jina"
    if kind == "chinese" or "chinese" in clip_name:
        return "chinese"
    return "clip"


def _loopfuse_class(backbone: str):
    if backbone == "chinese":
        from src.modeling_loopfuse_chinese import LoopFuse
    elif backbone == "jina":
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "src" / "modeling_loopfuse_jina-clip.py"
        spec = importlib.util.spec_from_file_location("modeling_loopfuse_jina_clip", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.LoopFuse
    else:
        from src.modeling_loopfuse import LoopFuse
    return LoopFuse


def _core_kwargs(config: dict) -> dict:
    num_heads_q = int(config.get("num_heads_q", config.get("n_heads", 8)))
    return {
        "clip_path": str(config["clip_name"]),
        "num_heads_q": num_heads_q,
        "num_heads_kv": int(config.get("num_heads_kv", num_heads_q)),
        "dropout_p": float(config.get("dropout", 0.1)),
        "num_classes": int(config.get("num_classes", 2)),
    }


def _build_core(config: dict, backbone: str) -> nn.Module:
    return _loopfuse_class(backbone)(**_core_kwargs(config))


def _adopt(module: nn.Module, core: nn.Module) -> None:
    """Move LoopFuse submodules onto `module` and drop the temporary owner."""
    for name in (
        "text_encoder",
        "vision_encoder",
        "text_projection",
        "visual_projection",
        "fuse",
        "classifier",
    ):
        child = getattr(core, name)
        delattr(core, name)
        setattr(module, name, child)
    module.projection_dim = core.projection_dim


def _hidden(output):
    if torch.is_tensor(output):
        return output
    last = getattr(output, "last_hidden_state", None)
    if last is not None:
        return last
    return output[0]


def _eos_pool(hidden: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    ids = input_ids.to(dtype=torch.int, device=hidden.device)
    if eos_token_id == 2:
        index = ids.argmax(dim=-1)
    else:
        index = (ids == eos_token_id).int().argmax(dim=-1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), index]


class _CLIPHeadMixin:
    baseline_name = "clip_head"
    forced_loops = 1

    def forward(self, input_ids, attention_mask, pixel_values, num_loops: int = 1, **kwargs):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            num_loops=1,
        )


class _SameHead(nn.Module):
    """Shared read-out: project every step, then the LoopFuse fusion head."""

    def __init__(self, config: dict, backbone: str, name: str, num_loops: int):
        super().__init__()
        if num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")
        self.baseline_name = name
        self.backbone = backbone
        self.num_loops = num_loops
        _adopt(self, _build_core(config, backbone))

    def _require_loops(self, num_loops: int) -> None:
        if num_loops != self.num_loops:
            raise ValueError(
                f"{self.baseline_name} was built with num_loops={self.num_loops}, got {num_loops}."
            )

    def _text_inputs(self, input_ids, attention_mask, position_ids, token_type_ids):
        encoder = self.text_encoder
        if self.backbone == "chinese":
            hidden = encoder.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
            mask = create_bidirectional_mask(
                config=encoder.config,
                inputs_embeds=hidden,
                attention_mask=attention_mask,
            )
            return hidden, mask
        hidden = encoder.embeddings(input_ids=input_ids, position_ids=position_ids)
        mask = create_causal_mask(
            config=encoder.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        return hidden, mask

    def _vision_inputs(self, pixel_values):
        hidden = self.vision_encoder.embeddings(pixel_values, interpolate_pos_encoding=False)
        return self.vision_encoder.pre_layrnorm(hidden)

    def _text_blocks(self):
        encoder = self.text_encoder.encoder
        return encoder.layer if self.backbone == "chinese" else encoder.layers

    def _run_text_block(self, block, hidden, mask):
        if self.backbone == "chinese":
            return _hidden(block(hidden, mask))
        return _hidden(block(hidden, mask, is_causal=True))

    def _run_text_encoder(self, encoder, hidden, mask):
        if self.backbone == "chinese":
            return _hidden(encoder(hidden, attention_mask=mask))
        return _hidden(encoder(inputs_embeds=hidden, attention_mask=mask, is_causal=True))

    def _run_vision_encoder(self, encoder, hidden):
        return _hidden(encoder(inputs_embeds=hidden))

    def _pool_text(self, hidden, input_ids):
        if self.backbone == "chinese":
            return hidden[:, 0]
        normalized = self.text_encoder.final_layer_norm(hidden)
        return _eos_pool(normalized, input_ids, self.text_encoder.eos_token_id)

    def _next_text(self, hidden):
        if self.backbone == "chinese":
            return hidden
        return self.text_encoder.final_layer_norm(hidden)

    def _pool_vision(self, hidden):
        return self.vision_encoder.post_layernorm(hidden[:, 0])

    def _logits(self, text_pools, vision_pools):
        text = [self.text_projection(item) for item in text_pools]
        vision = [self.visual_projection(item) for item in vision_pools]
        feature_t, feature_v = self.fuse(text, vision)
        return self.classifier(torch.cat([feature_t, feature_v], dim=-1))


class Stacked(_SameHead):
    """L non-shared copies of the full encoder, initialized from the checkpoint."""

    def __init__(self, config: dict, backbone: str, num_loops: int):
        super().__init__(config, backbone, "stacked", num_loops)
        text_encoder = self.text_encoder.encoder
        vision_encoder = self.vision_encoder.encoder
        self.text_stacks = nn.ModuleList(
            copy.deepcopy(text_encoder) for _ in range(num_loops - 1)
        )
        self.vision_stacks = nn.ModuleList(
            copy.deepcopy(vision_encoder) for _ in range(num_loops - 1)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        num_loops: int = 3,
        position_ids=None,
        token_type_ids=None,
    ):
        self._require_loops(num_loops)
        hidden, mask = self._text_inputs(input_ids, attention_mask, position_ids, token_type_ids)
        text_pools = []
        for encoder in (self.text_encoder.encoder, *self.text_stacks):
            hidden = self._run_text_encoder(encoder, hidden, mask)
            text_pools.append(self._pool_text(hidden, input_ids))
            hidden = self._next_text(hidden)

        hidden = self._vision_inputs(pixel_values)
        vision_pools = []
        for encoder in (self.vision_encoder.encoder, *self.vision_stacks):
            hidden = self._run_vision_encoder(encoder, hidden)
            vision_pools.append(self._pool_vision(hidden))
        return self._logits(text_pools, vision_pools)


class LayerTrajectory(_SameHead):
    """One forward pass. The trajectory is the last L layer read-outs."""

    def __init__(self, config: dict, backbone: str, num_loops: int):
        super().__init__(config, backbone, "layer_trajectory", num_loops)

    def _collect(self, blocks, hidden, step):
        depth = len(blocks)
        if self.num_loops > depth:
            raise ValueError(
                f"layer_trajectory asked for {self.num_loops} layers, but the tower has {depth}."
            )
        start = depth - self.num_loops
        pools = []
        for index, block in enumerate(blocks):
            hidden = step(block, hidden)
            if index >= start:
                pools.append(hidden)
        return pools

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        num_loops: int = 3,
        position_ids=None,
        token_type_ids=None,
    ):
        self._require_loops(num_loops)
        hidden, mask = self._text_inputs(input_ids, attention_mask, position_ids, token_type_ids)
        text_states = self._collect(
            self._text_blocks(),
            hidden,
            lambda block, current: self._run_text_block(block, current, mask),
        )
        text_pools = [self._pool_text(state, input_ids) for state in text_states]

        vision_states = self._collect(
            self.vision_encoder.encoder.layers,
            self._vision_inputs(pixel_values),
            lambda block, current: _hidden(block(current, None)),
        )
        vision_pools = [self._pool_vision(state) for state in vision_states]
        return self._logits(text_pools, vision_pools)


class Relaxed(_SameHead):
    """Shared whole-tower loop, plus a separate LoRA on attention at each loop."""

    def __init__(self, config: dict, backbone: str, num_loops: int):
        super().__init__(config, backbone, "relaxed", num_loops)
        rank = int(config.get("lora_rank", 8))
        alpha = float(config.get("lora_alpha", rank))
        linears = attention_linears(self.text_encoder, self.vision_encoder)
        self.lora = LoopLoRABank(linears, num_loops, rank, alpha)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        num_loops: int = 3,
        position_ids=None,
        token_type_ids=None,
    ):
        self._require_loops(num_loops)
        hidden, mask = self._text_inputs(input_ids, attention_mask, position_ids, token_type_ids)
        text_pools = []
        for index in range(self.num_loops):
            self.lora.set_loop(index)
            hidden = self._run_text_encoder(self.text_encoder.encoder, hidden, mask)
            text_pools.append(self._pool_text(hidden, input_ids))
            hidden = self._next_text(hidden)

        hidden = self._vision_inputs(pixel_values)
        vision_pools = []
        for index in range(self.num_loops):
            self.lora.set_loop(index)
            hidden = self._run_vision_encoder(self.vision_encoder.encoder, hidden)
            vision_pools.append(self._pool_vision(hidden))
        return self._logits(text_pools, vision_pools)


class _SinglePass(nn.Module):
    forced_loops = 1

    def __init__(self, config: dict, backbone: str, name: str):
        super().__init__()
        self.baseline_name = name
        self.backbone = backbone
        _adopt(self, _build_core(config, backbone))

    def _encode(self, input_ids, attention_mask, pixel_values):
        text = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, num_loops=1)
        vision = self.vision_encoder(pixel_values=pixel_values, num_loops=1)
        text = self.text_projection(text[-1].pooler_output)
        vision = self.visual_projection(vision[-1].pooler_output)
        return text, vision


class CLIPConcat(_SinglePass):
    """One pass, concatenate the two projected vectors, same MLP width as LoopFuse."""

    def __init__(self, config: dict, backbone: str):
        super().__init__(config, backbone, "clip_concat")
        self.fuse = nn.Identity()

    def forward(self, input_ids, attention_mask, pixel_values, num_loops: int = 1, **kwargs):
        text, vision = self._encode(input_ids, attention_mask, pixel_values)
        return self.classifier(torch.cat([text, vision], dim=-1))


class CLIPInteract(_SinglePass):
    """One pass. Product, absolute difference, and cosine join the two vectors."""

    def __init__(self, config: dict, backbone: str):
        super().__init__(config, backbone, "clip_interact")
        dropout_p = float(config.get("dropout", 0.1))
        width = self.projection_dim
        self.fuse = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(width * 4 + 1, width),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(width, int(config.get("num_classes", 2))),
        )

    def forward(self, input_ids, attention_mask, pixel_values, num_loops: int = 1, **kwargs):
        text, vision = self._encode(input_ids, attention_mask, pixel_values)
        text_n = F.normalize(text, dim=-1)
        vision_n = F.normalize(vision, dim=-1)
        cosine = (text_n * vision_n).sum(dim=-1, keepdim=True)
        features = torch.cat([text, vision, text * vision, (text - vision).abs(), cosine], dim=-1)
        return self.classifier(features)


def build_clip_head(config: dict, backbone: str) -> nn.Module:
    base = _loopfuse_class(backbone)
    head_cls = type(
        "ChineseCLIPHead" if backbone == "chinese" else "CLIPHead",
        (_CLIPHeadMixin, base),
        {},
    )
    return head_cls(**_core_kwargs(config))
