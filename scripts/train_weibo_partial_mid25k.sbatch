#!/bin/bash
#SBATCH --job-name=weibo-mid25k
#SBATCH --account=lumia
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --nodelist=node2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs
export NPROC=4
export SPLIT=mid25k
export SEEDS="0 1 2 3 4 5 6 7"
export MASTER_PORT=29841
export OUT=outputs/weibo_partial_mid25k
export INIT_CKPT=/data3/yxfeng/experiments/cnclip_midtrain_l2_gate/step_25000.pt
bash scripts/run_weibo_partial_loop.sh
