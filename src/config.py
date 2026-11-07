"""Flat experiment config, loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class CSCWDConfig:
    clip_name: str = "openai/clip-vit-base-patch32"
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.2
    ffn_ratio: int = 4
    freeze_clip: bool = True
    use_fusion: bool = True
    use_stage1: bool = True
    use_stage2: bool = True
    num_classes: int = 2

    train_jsonl: str = "data/weibo/train.jsonl"
    val_jsonl: str = ""
    test_jsonl: str = "data/weibo/test.jsonl"
    image_root: str = "data/weibo_raw"
    val_ratio: float = 0.1
    max_text_length: int = 77
    batch_size: int = 128
    num_workers: int = 0

    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    seed: int = 42
    output_dir: str = "outputs/weibo"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "CSCWDConfig":
        defaults = {item.name: item.default for item in fields(cls)}
        unknown = sorted(set(raw) - set(defaults))
        if unknown:
            raise SystemExit(f"Unknown config keys: {', '.join(unknown)}")
        config = cls()
        for key, value in raw.items():
            setattr(config, key, _coerce_value(key, value, defaults[key]))
        return config


def _coerce_value(key: str, value: object, default: object) -> object:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        raise SystemExit(f"Config key {key!r} must be a boolean, got {value!r}")
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, bool):
            raise SystemExit(f"Config key {key!r} must be an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
        raise SystemExit(f"Config key {key!r} must be an integer, got {value!r}")
    if isinstance(default, float):
        if isinstance(value, bool):
            raise SystemExit(f"Config key {key!r} must be a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise SystemExit(f"Config key {key!r} must be a number, got {value!r}")
    if isinstance(default, str):
        if value is None:
            return ""
        if not isinstance(value, str):
            raise SystemExit(f"Config key {key!r} must be a string, got {value!r}")
        return value
    return value


def load_config(path: str | None) -> CSCWDConfig:
    config = CSCWDConfig()
    if not path:
        return config
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return CSCWDConfig.from_dict(raw)
