#!/bin/bash
#SBATCH --job-name=weibo-in4k-both
#SBATCH --account=lumia
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:4
#SBATCH --mem=128G
#SBATCH --nodelist=node3
#SBATCH --time=08:00:00
#SBATCH --output=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.err

# Both towers, loop=2, eight seeds. Init from the ImageNet loop=2 step_4000 checkpoint.
set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs

export NPROC=4
export SEEDS="0 1 2 3 4 5 6 7"
export MODEL_TYPE=partial_loop
export LOOPS=2
export INIT_CKPT=/data3/yxfeng/experiments/cnclip_midtrain_l2_imagenet_zh/step_4000.pt
export LOOP_TEXT=true
export LOOP_VISION=true
export SPLIT=imagenet4k-both
export OUT=outputs/weibo_partial_imagenet4k_both
export LOG_TAG=weibo_partial_imagenet4k_both
export MASTER_PORT=29891

bash scripts/run_weibo_partial_loop.sh
