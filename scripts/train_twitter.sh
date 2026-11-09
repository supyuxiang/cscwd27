#!/bin/bash
cd "$(dirname "$0")/.."
python train_dp.py --config configs/twitter.yaml
