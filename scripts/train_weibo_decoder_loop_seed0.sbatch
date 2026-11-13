#!/bin/bash
#SBATCH --job-name=weibo-loopdec
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
export SPLIT=decoder-loop
export SEEDS="0"
export MASTER_PORT=29847
export OUT=outputs/weibo_decoder_loop
export EVAL_TEST_EACH_EPOCH=1
export MODEL_TYPE=decoder_loop
bash scripts/run_weibo_partial_loop.sh
