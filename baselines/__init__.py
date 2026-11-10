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
}


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
