#!/bin/bash
#SBATCH --job-name=weibo-direct
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodelist=node6
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
bash scripts/run_weibo_directfuse.sh
