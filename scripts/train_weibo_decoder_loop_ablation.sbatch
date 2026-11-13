#!/bin/bash
#SBATCH --job-name=weibo-dloop
#SBATCH --account=lumia
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --nodelist=node2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs
export NPROC=4
export SEEDS="0"
export MODEL_TYPE=decoder_loop
export EVAL_TEST_EACH_EPOCH=1
export LOOPS="${LOOPS:?set LOOPS}"
export SPLIT="decoder-l${LOOPS}"
export OUT="outputs/weibo_decoder_loop_l${LOOPS}"
export LOG_TAG="weibo_decoder_l${LOOPS}"
export MASTER_PORT="${MASTER_PORT:?set MASTER_PORT}"
bash scripts/run_weibo_partial_loop.sh
