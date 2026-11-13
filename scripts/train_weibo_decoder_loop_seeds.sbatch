#!/bin/bash
#SBATCH --job-name=weibo-dseed
#SBATCH --account=lumia
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --nodelist=node2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs
export NPROC=4
export MODEL_TYPE=decoder_loop
export LOOPS="${LOOPS:-2}"
export OUT="${OUT:-outputs/weibo_decoder_skip_l${LOOPS}}"
export SEEDS="${SEED_LIST//-/ }"
export SPLIT="decoder-skip-l${LOOPS}-${SPLIT_TAG:?set SPLIT_TAG}"
export LOG_TAG="weibo_decoder_skip_l${LOOPS}_${SPLIT_TAG}"
export MASTER_PORT="${MASTER_PORT:?set MASTER_PORT}"
bash scripts/run_weibo_partial_loop.sh
