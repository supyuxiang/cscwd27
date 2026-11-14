#!/bin/bash
#SBATCH --job-name=cnclip-mid-l4
#SBATCH --account=lumia
#SBATCH --partition=ADA6000
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:2
#SBATCH --mem=200G
#SBATCH --nodelist=node9
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.err

# Loop count is fixed at 4. Weights come from the loop=2 mid-train checkpoint.
# The loss is the last loop only. Two ADA6000s.
# Micro-batch 32, accumulated 4 times, so the contrastive batch is 32 × 2 × 4 = 256.
# 23438 steps cover 6,000,128 samples, the same two passes as the previous run.

set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs

python3 - << 'PY'
import json
from pathlib import Path
counts = json.loads(Path("/data3/yxfeng/datasets/cnclip_pretrain/counts.json").read_text())
if counts["total"] < 3_000_000:
    raise SystemExit(f"image count is short: {counts}")
print("images", counts, flush=True)
PY

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MASTER_PORT=29851

OUT=${CNCLIP_MID_OUT:-/data3/yxfeng/experiments/cnclip_midtrain_l4_lr5e-5}
INIT=${CNCLIP_MID_INIT:-/data3/yxfeng/experiments/cnclip_midtrain_l2_loop1loss/latest.pt}
mkdir -p "$OUT"
TORCHRUN=/home/yxfeng/miniconda3/envs/www27/bin/torchrun
DATA=/data3/yxfeng/datasets/cnclip_pretrain/jsonl
CLIP=/data3/yxfeng/models/clip/chinese-clip-vit-base-patch16

run_train() {
    local batch="$1"
    local accum="$2"
    local steps="$3"
    local global_batch=$((batch * 2 * accum))
    if [[ "$global_batch" -ne 256 ]]; then
        echo "global batch ${global_batch} is not 256" >&2
        exit 1
    fi
    local extra=()
    if [[ -f "$OUT/latest.pt" ]]; then
        extra=(--resume "$OUT/latest.pt")
    else
        extra=(--init "$INIT")
    fi
    "$TORCHRUN" --nproc_per_node=2 --master_port="$MASTER_PORT" -m src.mid_training \
        --clip "$CLIP" \
        --data "$DATA" \
        --output "$OUT" \
        --steps "$steps" \
        --warmup-steps 0 \
        --lr-warmup 2000 \
        --batch-size "$batch" \
        --grad-accum "$accum" \
        --lr 5e-5 \
        --weight-decay 0.01 \
        --num-loops 4 \
        --max-length 512 \
        --num-workers 4 \
        --log-every 20 \
        --save-every 5000 \
        "${extra[@]}"
}

if ! run_train 32 4 23438; then
    if grep -qi "out of memory" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"; then
        echo "micro-batch 32 ran out of memory, retrying at micro-batch 16 with 8 accumulations" >&2
        run_train 16 8 23438
    else
        exit 1
    fi
fi
