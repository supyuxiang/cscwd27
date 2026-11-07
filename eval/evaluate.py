"""Evaluate a trained CMAAN checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CSCWDConfig, load_config
from src.data import MultimodalDataset, build_collate, load_jsonl_samples
from src.model import CMAAN
from src.utils import load_checkpoint, run_epoch_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CMAAN on a JSONL split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Override data paths. Model flags stay in the checkpoint.")
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = CSCWDConfig.from_dict(checkpoint["config"])
    if args.config is not None:
        override = load_config(str(args.config))
        config.train_jsonl = override.train_jsonl
        config.val_jsonl = override.val_jsonl
        config.test_jsonl = override.test_jsonl
        config.image_root = override.image_root
        config.max_text_length = override.max_text_length
        config.batch_size = override.batch_size
        config.num_workers = override.num_workers
    if args.jsonl is not None:
        config.test_jsonl = str(args.jsonl)
    if args.image_root is not None:
        config.image_root = args.image_root
    jsonl_path = config.test_jsonl.strip() or config.val_jsonl.strip()
    if not jsonl_path:
        raise SystemExit("Pass --jsonl or set test_jsonl in the checkpoint config.")

    samples, missing = load_jsonl_samples(jsonl_path, config.image_root or None)
    model = CMAAN.from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])
    batch_size = args.batch_size or int(config.batch_size)
    loader = DataLoader(
        MultimodalDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.num_workers),
        collate_fn=build_collate(model.processor, int(config.max_text_length)),
    )
    metrics, report = run_epoch_eval(model, loader, device)
    print(f"samples: {len(samples)} (missing images {missing})")
    print(json.dumps(metrics, indent=2))
    print(report)


if __name__ == "__main__":
    main()
