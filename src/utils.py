from __future__ import annotations

import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from .data import LoopFuseDataset, load_jsonl_samples, loopfuse_collator, stratified_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def prepare_splits(config: dict) -> dict[str, list | dict]:
    """Load JSONL manifests. An empty val file falls back to a stratified split of train."""
    image_root = config.get("image_root") or None
    train_samples, train_missing = load_jsonl_samples(config["train_jsonl"], image_root)
    val_path = str(config.get("val_jsonl") or "").strip()
    val_missing = 0
    if val_path:
        val_samples, val_missing = load_jsonl_samples(val_path, image_root)
    else:
        train_samples, val_samples = stratified_split(
            train_samples,
            float(config.get("val_ratio", 0.0)),
            int(config.get("seed", 42)),
        )
    test_path = str(config.get("test_jsonl") or "").strip()
    test_samples: list = []
    test_missing = 0
    if test_path:
        test_samples, test_missing = load_jsonl_samples(test_path, image_root)
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "missing": {"train": train_missing, "val": val_missing, "test": test_missing},
    }


def label_counts(samples: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for sample in samples:
        label = int(sample["label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def make_loader(
    dataset: LoopFuseDataset,
    processor,
    config: dict,
    *,
    shuffle: bool,
    sampler: DistributedSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=loopfuse_collator(processor, int(config["max_text_length"])),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_dataloaders(
    config: dict,
    processor,
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    is_main: bool = True,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, DistributedSampler | None]:
    splits = prepare_splits(config)
    if is_main:
        missing = splits["missing"]
        for name in ("train", "val", "test"):
            samples = splits[name]
            if not samples and name == "test":
                continue
            print(
                f"{name}: {len(samples)} samples {label_counts(samples)} "
                f"missing_images={missing[name]}"
            )

    train_set = LoopFuseDataset(splits["train"])
    train_sampler = (
        DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config.get("seed", 42)),
        )
        if distributed
        else None
    )
    train_loader = make_loader(
        train_set, processor, config, shuffle=True, sampler=train_sampler
    )
    val_loader = (
        make_loader(LoopFuseDataset(splits["val"]), processor, config, shuffle=False)
        if splits["val"] and is_main
        else None
    )
    test_loader = (
        make_loader(LoopFuseDataset(splits["test"]), processor, config, shuffle=False)
        if splits["test"] and is_main
        else None
    )
    return train_loader, val_loader, test_loader, train_sampler


def masked_mean(tokens: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
    """Average over the token axis. Pad positions are ignored."""
    if pad is None:
        return tokens.mean(dim=1)
    valid = (~pad).unsqueeze(-1).to(dtype=tokens.dtype)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (tokens * valid).sum(dim=1) / denom


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _binary_counts(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    if len(y_true) != len(y_pred):
        raise ValueError(f"Prediction length {len(y_pred)} does not match labels {len(y_true)}.")
    matrix = [[0, 0], [0, 0]]
    for truth, pred in zip(y_true, y_pred):
        if truth not in (0, 1) or pred not in (0, 1):
            raise ValueError(f"Expected binary labels, got true={truth} pred={pred}.")
        matrix[truth][pred] += 1
    return matrix


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision_fake": 0.0,
            "recall_fake": 0.0,
            "f1_fake": 0.0,
            "precision_real": 0.0,
            "recall_real": 0.0,
            "f1_real": 0.0,
            "macro_f1": 0.0,
        }
    matrix = _binary_counts(y_true, y_pred)
    real_p, real_r, real_f = _prf(matrix[0][0], matrix[1][0], matrix[0][1])
    fake_p, fake_r, fake_f = _prf(matrix[1][1], matrix[0][1], matrix[1][0])
    correct = matrix[0][0] + matrix[1][1]
    return {
        "accuracy": correct / len(y_true),
        "precision_fake": fake_p,
        "recall_fake": fake_r,
        "f1_fake": fake_f,
        "precision_real": real_p,
        "recall_real": real_r,
        "f1_real": real_f,
        "macro_f1": (real_f + fake_f) / 2.0,
    }


def report_predictions(y_true: list[int], y_pred: list[int]) -> str:
    scores = score_predictions(y_true, y_pred)
    support_real = sum(label == 0 for label in y_true)
    support_fake = sum(label == 1 for label in y_true)

    def row(name: str, precision: float, recall: float, f1: float, support: int) -> str:
        return f"{name:>12} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}"

    macro_p = (scores["precision_real"] + scores["precision_fake"]) / 2.0
    macro_r = (scores["recall_real"] + scores["recall_fake"]) / 2.0
    lines = [
        f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
        "",
        row("real", scores["precision_real"], scores["recall_real"], scores["f1_real"], support_real),
        row("fake", scores["precision_fake"], scores["recall_fake"], scores["f1_fake"], support_fake),
        "",
        f"{'accuracy':>12} {'':>10} {'':>10} {scores['accuracy']:10.4f} {len(y_true):10d}",
        row("macro avg", macro_p, macro_r, scores["macro_f1"], len(y_true)),
    ]
    return "\n".join(lines)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def run_epoch_eval(model, loader, device, num_loops: int = 3) -> tuple[dict[str, float], str]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            num_loops=num_loops,
        )
        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return score_predictions(y_true, y_pred), report_predictions(y_true, y_pred)
