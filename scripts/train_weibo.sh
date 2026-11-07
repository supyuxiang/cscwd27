#!/bin/bash
cd "$(dirname "$0")/.."
python eval/train.py --config configs/weibo.yaml
