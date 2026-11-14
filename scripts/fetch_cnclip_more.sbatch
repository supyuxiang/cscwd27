#!/bin/bash
#SBATCH --job-name=cnclip-fetch-more
#SBATCH --account=lumia
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.err

# Add 2,000,000 images to the existing 3,000,189. Longer captions are tried first.
set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs
export PYTHONUNBUFFERED=1
/home/yxfeng/miniconda3/envs/verl1/bin/python scripts/prepare_cnclip_midtrain.py \
    --root /data3/yxfeng/datasets/cnclip_pretrain \
    --target 5000000 \
    --min-chars 51 \
    --fill-chars 40 \
    --workers 256 \
    --timeout 3
