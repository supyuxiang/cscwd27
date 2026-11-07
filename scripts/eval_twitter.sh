#!/bin/bash
cd "$(dirname "$0")/.."
python eval/evaluate.py --checkpoint outputs/twitter/best.pt --config configs/twitter.yaml
