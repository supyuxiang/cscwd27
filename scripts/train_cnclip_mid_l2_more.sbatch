#!/bin/bash
#SBATCH --job-name=cnclip-mid-l2
#SBATCH --account=lumia
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:4
#SBATCH --mem=200G
#SBATCH --nodelist=node3
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.err

# Fresh loop=2 mid-train. Last-loop loss. Global contrastive batch 256.
# Three epochs over the expanded Chinese pretrain set. Checkpoint and Flickr eval every 2000 steps.
set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MASTER_PORT=29871

OUT=/data3/yxfeng/experiments/cnclip_midtrain_l2_lr2e-5
mkdir -p "$OUT"
TORCHRUN=/home/yxfeng/miniconda3/envs/www27/bin/torchrun
DATA=/data3/yxfeng/datasets/cnclip_pretrain/jsonl
CLIP=/data3/yxfeng/models/clip/chinese-clip-vit-base-patch16

STEPS=$(/home/yxfeng/miniconda3/envs/www27/bin/python - << 'PY'
import json, math
from pathlib import Path
counts = json.loads(Path("/data3/yxfeng/datasets/cnclip_pretrain/counts.json").read_text())
total = int(counts["total"])
if total < 4_800_000:
    raise SystemExit(f"need at least 4800000 images, have {total}")
steps = math.ceil(total * 3 / 256)
print(steps)
PY
)
echo "images-to-steps ${STEPS}"

run_train() {
    local batch="$1"
    local accum="$2"
    local global_batch=$((batch * 4 * accum))
    if [[ "$global_batch" -ne 256 ]]; then
        echo "global batch ${global_batch} is not 256" >&2
        exit 1
    fi
    local extra=()
    if [[ -f "$OUT/latest.pt" ]]; then
        extra=(--resume "$OUT/latest.pt")
    fi
    "$TORCHRUN" --nproc_per_node=4 --master_port="$MASTER_PORT" -m src.mid_training \
        --clip "$CLIP" \
        --data "$DATA" \
        --output "$OUT" \
        --steps "$STEPS" \
        --warmup-steps 0 \
        --lr-warmup 2000 \
        --batch-size "$batch" \
        --grad-accum "$accum" \
        --lr 2e-5 \
        --weight-decay 0.01 \
        --num-loops 2 \
        --max-length 512 \
        --num-workers 4 \
        --log-every 20 \
        --save-every 2000 \
        "${extra[@]}"
}

if ! run_train 64 1; then
    if grep -qi "out of memory" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"; then
        echo "batch 64 ran out of memory, retrying at batch 32 x accum 2" >&2
        run_train 32 2
    else
        exit 1
    fi
fi
