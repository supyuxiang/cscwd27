"""Train CMAAN and evaluate the best checkpoint on the official test split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import MultimodalDataset, build_collate, load_jsonl_samples, stratified_split
from src.model import CMAAN
from src.utils import load_checkpoint, move_batch, run_epoch_eval


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, collate, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    drop_last = shuffle and len(dataset) > batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
    )


def save_checkpoint(path: Path, model: CMAAN, config: dict, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def run_epoch_train(model, loader, optimizer, device) -> float:
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_steps = 0
    progress = tqdm(loader, leave=False)
    for batch in progress:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        total_steps += 1
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    return total_loss / max(total_steps, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMAAN for multimodal fake news detection.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(str(args.config))
    set_seed(int(config.seed))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    image_root = config.image_root or None
    train_samples, train_missing = load_jsonl_samples(config.train_jsonl, image_root)
    if config.val_jsonl.strip():
        val_samples, val_missing = load_jsonl_samples(config.val_jsonl, image_root)
    else:
        train_samples, val_samples = stratified_split(train_samples, float(config.val_ratio), int(config.seed))
        val_missing = 0
    test_samples = None
    test_missing = 0
    if config.test_jsonl.strip():
        test_samples, test_missing = load_jsonl_samples(config.test_jsonl, image_root)

    print(f"device: {device}")
    print(f"train: {len(train_samples)} (missing images {train_missing})")
    if val_samples:
        print(f"val:   {len(val_samples)} (missing images {val_missing})")
    if test_samples is not None:
        print(f"test:  {len(test_samples)} (missing images {test_missing})")

    model = CMAAN.from_config(config).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters: {trainable:,} trainable / {total:,} total")

    collate = build_collate(model.processor, int(config.max_text_length))
    batch_size = int(config.batch_size)
    num_workers = int(config.num_workers)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        MultimodalDataset(train_samples), collate, batch_size, True, num_workers, pin_memory
    )
    val_loader = None
    if val_samples:
        val_loader = make_loader(
            MultimodalDataset(val_samples), collate, batch_size, False, num_workers, pin_memory
        )

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    epochs = int(config.epochs)
    patience = int(config.patience)
    best_score = (-1.0, -1.0)
    stale = 0
    best_epoch = 0
    epoch = 0
    history = []
    stored = config.to_dict()
    for epoch in range(1, epochs + 1):
        loss = run_epoch_train(model, train_loader, optimizer, device)
        record = {"epoch": epoch, "train_loss": loss}
        message = f"epoch {epoch:02d}  loss {loss:.4f}"
        if val_loader is not None:
            metrics, _report = run_epoch_eval(model, val_loader, device)
            record["val"] = metrics
            message += (
                f"  val_acc {metrics['accuracy']:.4f}"
                f"  val_f1_fake {metrics['f1_fake']:.4f}"
                f"  val_macro_f1 {metrics['macro_f1']:.4f}"
            )
            score = (metrics["accuracy"], metrics["macro_f1"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale = 0
                save_checkpoint(output_dir / "best.pt", model, stored, epoch, metrics)
            else:
                stale += 1
        else:
            save_checkpoint(output_dir / "best.pt", model, stored, epoch, {})
        history.append(record)
        print(message)
        if val_loader is not None and patience > 0 and stale >= patience:
            print(f"early stop at epoch {epoch}; best epoch {best_epoch}")
            break

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    if history:
        save_checkpoint(output_dir / "last.pt", model, stored, epoch, history[-1])
    if test_samples is not None and (output_dir / "best.pt").is_file():
        checkpoint = load_checkpoint(output_dir / "best.pt", device)
        model.load_state_dict(checkpoint["model"])
        test_loader = make_loader(
            MultimodalDataset(test_samples), collate, batch_size, False, num_workers, pin_memory
        )
        metrics, report = run_epoch_eval(model, test_loader, device)
        (output_dir / "test_report.txt").write_text(report, encoding="utf-8")
        (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("test set, best checkpoint")
        print(report)


if __name__ == "__main__":
    main()
