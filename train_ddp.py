"""
torchrun --nproc_per_node=4 train_ddp.py --config configs/weibo.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from train import init_distributed, load_experiment, run_training
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Full finetune with DDP")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    rank, world_size, local_rank, device = init_distributed()
    is_main = rank == 0
    try:
        config = load_config(args.config)
        model, train_loader, val_loader, test_loader, train_sampler = load_experiment(
            config,
            distributed=True,
            rank=rank,
            world_size=world_size,
            is_main=is_main,
        )
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        run_training(
            config,
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
            train_sampler=train_sampler,
            is_main=is_main,
            distributed=True,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
