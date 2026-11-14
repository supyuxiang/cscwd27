#!/bin/bash
#SBATCH --job-name=weibo-pl
#SBATCH --account=lumia
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
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
if [[ -n "${SEED_LIST:-}" ]]; then
  export SEEDS="${SEED_LIST//-/ }"
else
  export SEEDS="0"
fi
export MODEL_TYPE=partial_loop
export LOOPS=2
export INIT_CKPT=/data3/yxfeng/experiments/cnclip_midtrain_l2_loop1loss/latest.pt
export TOWER="${TOWER:?set TOWER to text, vision, or both}"
case "$TOWER" in
  text) export LOOP_TEXT=true; export LOOP_VISION=false ;;
  vision) export LOOP_TEXT=false; export LOOP_VISION=true ;;
  both) export LOOP_TEXT=true; export LOOP_VISION=true ;;
  *) echo "unknown TOWER=$TOWER" >&2; exit 1 ;;
esac
export SPLIT="partial-mid-${TOWER}"
export OUT="outputs/weibo_partial_mid_${TOWER}"
export LOG_TAG="weibo_partial_mid_${TOWER}"
export MASTER_PORT="${MASTER_PORT:?set MASTER_PORT}"
bash scripts/run_weibo_partial_loop.sh
