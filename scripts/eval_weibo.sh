#!/bin/bash
cd "$(dirname "$0")/.."
python eval/evaluate.py --checkpoint outputs/weibo/best.pt --config configs/weibo.yaml
