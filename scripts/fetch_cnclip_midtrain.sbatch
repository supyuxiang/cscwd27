#!/bin/bash
#SBATCH --job-name=cnclip-fetch
#SBATCH --account=lumia
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs
export PYTHONUNBUFFERED=1
/home/yxfeng/miniconda3/envs/verl1/bin/python scripts/prepare_cnclip_midtrain.py \
    --root /data3/yxfeng/datasets/cnclip_pretrain \
    --target 3000000 \
    --min-chars 51 \
    --fill-chars 40 \
    --workers 640 \
    --timeout 3
