#!/bin/bash
#SBATCH --job-name=weibo-lora
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
export SPLIT=bothloop-lora-lr1e4
export SEEDS="0"
export MASTER_PORT=29846
export OUT=outputs/weibo_partial_mid25k_bothloop_lora_lr1e-4
export INIT_CKPT=/data3/yxfeng/experiments/cnclip_midtrain_l2_gate/step_25000.pt
export EVAL_TEST_EACH_EPOCH=1
export LR=1e-4
export MIN_LR=1e-5
export LORA_CLIP=1
export LORA_RANK=8
export LORA_ALPHA=8
bash scripts/run_weibo_partial_loop.sh
