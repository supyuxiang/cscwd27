#!/bin/bash
#SBATCH --job-name=cnclip-eval-l4
#SBATCH --account=lumia
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.err

set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
export PYTHONUNBUFFERED=1
: "${CKPT:?set CKPT}"
: "${LABEL:?set LABEL}"
: "${CNCLIP_MID_OUT:?set CNCLIP_MID_OUT}"
PY=/home/yxfeng/miniconda3/envs/www27/bin/python
METRICS="$CNCLIP_MID_OUT/eval/step_${LABEL}.json"

echo "===== $(date '+%F %T') step ${LABEL} ====="
"$PY" scripts/eval_cnclip_retrieval.py \
    --checkpoint "$CKPT" \
    --num-loops 4 \
    --batch-size 32 \
    --max-length 128 \
    --metrics-json "$METRICS"
echo "EVAL step ${LABEL} done"
