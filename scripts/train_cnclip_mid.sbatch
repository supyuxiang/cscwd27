#!/bin/bash
#SBATCH --job-name=cnclip-mid
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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
export MASTER_PORT=29831

OUT=/data3/yxfeng/experiments/cnclip_midtrain_l2_gate
mkdir -p "$OUT"
TORCHRUN=/home/yxfeng/miniconda3/envs/www27/bin/torchrun
DATA=/data3/yxfeng/datasets/cnclip_pretrain/jsonl
CLIP=/data3/yxfeng/models/clip/chinese-clip-vit-base-patch16

run_train() {
    local batch="$1"
    local steps="$2"
    local resume=()
    if [[ -f "$OUT/latest.pt" ]]; then
        resume=(--resume "$OUT/latest.pt")
    fi
    "$TORCHRUN" --nproc_per_node=4 --master_port="$MASTER_PORT" -m src.mid_training \
        --clip "$CLIP" \
        --data "$DATA" \
        --output "$OUT" \
        --steps "$steps" \
        --warmup-steps 2000 \
        --lr-warmup 2000 \
        --batch-size "$batch" \
        --lr 5e-6 \
        --weight-decay 0.01 \
        --num-loops 2 \
        --max-length 512 \
        --distill-weight 10 \
        --distill-weight-end 0.1 \
        --num-workers 6 \
        --log-every 20 \
        --save-every 5000 \
        "${resume[@]}"
}

if ! run_train 32 46875; then
    if grep -qi "out of memory" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"; then
        echo "batch 32 ran out of memory, retrying at batch 16" >&2
        run_train 16 93750
    else
        exit 1
    fi
fi
