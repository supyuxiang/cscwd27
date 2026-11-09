from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.nn import DataParallel

from train import load_experiment, run_training
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Full finetune with DataParallel")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_loader, val_loader, test_loader, _sampler = load_experiment(
        config,
        distributed=False,
        rank=0,
        world_size=1,
        is_main=True,
    )
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f"DataParallel on {torch.cuda.device_count()} GPUs")
    run_training(
        config,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        is_main=True,
        distributed=False,
    )


if __name__ == "__main__":
    main()
