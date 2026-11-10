#!/bin/bash
cd "$(dirname "$0")/.."
torchrun --nproc_per_node=4 trainer.py --config=weibo
