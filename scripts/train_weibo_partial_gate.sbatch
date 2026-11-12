#!/bin/bash
#SBATCH --job-name=weibo-partial
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodelist=node7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs
export NPROC=4
export SPLIT=all
export SEEDS="0 1 2 3 4 5 6 7"
export MASTER_PORT=29821
bash scripts/run_weibo_partial_loop.sh
