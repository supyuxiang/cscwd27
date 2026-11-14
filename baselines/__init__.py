"""Same-backbone detectors that train through trainer.py."""

from .models import (
    CLIPConcat,
    CLIPInteract,
    LayerTrajectory,
    Relaxed,
    Stacked,
    build_clip_head,
    resolve_backbone,
)

_NAMES = {
    "clip": "loopfuse",
    "chinese": "loopfuse",
    "loopfuse": "loopfuse",
    "jina": "loopfuse",
    "jina-clip": "loopfuse",
    "jinaclip": "loopfuse",
    "clip_head": "clip_head",
    "clip-head": "clip_head",
    "stacked": "stacked",
    "layer_trajectory": "layer_trajectory",
    "layer-trajectory": "layer_trajectory",
    "relaxed": "relaxed",
    "clip_concat": "clip_concat",
    "clip-concat": "clip_concat",
    "clip_interact": "clip_interact",
    "clip-interact": "clip_interact",
    "loopback": "loopback",
    "loopprobe": "loopprobe",
    "directfuse": "directfuse",
    "decoder_loop": "decoder_loop",
    "decoder-loop": "decoder_loop",
    "partial_loop": "partial_loop",
    "randproj": "randproj",
}


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def build_detector(config: dict):
    """Build LoopFuse or a same-backbone baseline from a trainer config."""
    raw = str(config.get("model_type", "clip"))
    name = _NAMES.get(raw)
    if name is None:
        choices = ", ".join(sorted(set(_NAMES)))
        raise ValueError(f"Unknown model_type {raw!r}. Choose one of: {choices}.")
    backbone = resolve_backbone(config, raw)
    if name == "loopfuse":
        from .models import _build_core

        model = _build_core(config, backbone)
        model.baseline_name = "loopfuse"
        return model
    if name == "loopback":
        if backbone != "chinese":
            raise ValueError("loopback is implemented for the Chinese CLIP backbone.")
        from .models import _core_kwargs
        from src.modeling_loopfuse_chinese import Loopback

        model = Loopback(**_core_kwargs(config))
        model.baseline_name = "loopback"
        return model
    if name == "loopprobe":
        if backbone != "chinese":
            raise ValueError("loopprobe is implemented for the Chinese CLIP backbone.")
        from .models import _core_kwargs
        from src.modeling_loopfuse_chinese import LoopProbe

        model = LoopProbe(**_core_kwargs(config))
        model.baseline_name = "loopprobe"
        return model
    if name == "directfuse":
        if backbone != "chinese":
            raise ValueError("directfuse is implemented for the Chinese CLIP backbone.")
        from .models import _core_kwargs
        from src.modeling_loopfuse_chinese import DirectFuse

        model = DirectFuse(**_core_kwargs(config))
        model.baseline_name = "directfuse"
        return model
    if name == "decoder_loop":
        if backbone != "chinese":
            raise ValueError("decoder_loop is implemented for the Chinese CLIP backbone.")
        from .models import _core_kwargs
        from src.modeling_loopfuse_chinese import DecoderLoop

        model = DecoderLoop(**_core_kwargs(config))
        model.baseline_name = "decoder_loop"
        return model
    if name == "partial_loop":
        if backbone != "chinese":
            raise ValueError("partial_loop is implemented for the Chinese CLIP backbone.")
        from .models import _core_kwargs
        from src.modeling_loopfuse_chinese import LoopFuse_partial_loop

        model = LoopFuse_partial_loop(**_core_kwargs(config))
        model.loop_text = _as_bool(config.get("loop_text", True))
        model.loop_vision = _as_bool(config.get("loop_vision", True))
        if _as_bool(config.get("lora_clip", False)):
            from baselines.lora import LoopLoRABank, attention_linears

            rank = int(config.get("lora_rank", 8))
            alpha = float(config.get("lora_alpha", rank))
            linears = attention_linears(model.text_encoder, model.vision_encoder)
            if not linears:
                raise RuntimeError("lora_clip found no attention projections on the Chinese CLIP towers.")
            model.lora = LoopLoRABank(linears, num_loops=1, rank=rank, alpha=alpha)
            model.lora_linear_count = len(linears)
            model.lora_rank = rank
            model.lora_alpha = alpha
        model.baseline_name = "partial_loop"
        return model
    if name == "randproj":
        if backbone != "chinese":
            raise ValueError("randproj is implemented for the Chinese CLIP backbone.")
        from .models import _core_kwargs
        from src.modeling_loopfuse_chinese import LoopFuseRandProj

        model = LoopFuseRandProj(**_core_kwargs(config))
        model.baseline_name = "randproj"
        return model

    num_loops = int(config.get("num_loops", 3))
    if name == "clip_head":
        return build_clip_head(config, backbone)
    if name == "stacked":
        return Stacked(config, backbone, num_loops)
    if name == "layer_trajectory":
        return LayerTrajectory(config, backbone, num_loops)
    if name == "relaxed":
        return Relaxed(config, backbone, num_loops)
    if name == "clip_concat":
        return CLIPConcat(config, backbone)
    return CLIPInteract(config, backbone)


__all__ = ["build_detector"]
