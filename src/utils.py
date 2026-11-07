"""Pooling, checkpoints, and classification metrics."""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def masked_mean(tokens: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
    """Adaptive average pool over the token axis. Pad positions are ignored."""
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


def score_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_real": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_real": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_real": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def report_predictions(y_true: list[int], y_pred: list[int]) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["real", "fake"],
        digits=4,
        zero_division=0,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def run_epoch_eval(model, loader, device) -> tuple[dict[str, float], str]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        prediction = logits.argmax(dim=-1)
        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(prediction.detach().cpu().tolist())
    return score_predictions(y_true, y_pred), report_predictions(y_true, y_pred)
