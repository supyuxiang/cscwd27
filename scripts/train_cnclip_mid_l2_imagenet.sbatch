#!/bin/bash
#SBATCH --job-name=cnclip-mid-l2in
#SBATCH --account=lumia
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:8
#SBATCH --mem=400G
#SBATCH --nodelist=node2
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.err

# Second loop=2 mid-train round. Weights come from the LAION loop=2 checkpoint.
# Data is ImageNet train with Chinese-CLIP class names and photo prompts.
# High QoS allows 4 GPUs, so these 8 cards use normal.
# 10010 steps * batch 32 * 8 GPUs = 2,562,560 samples, two passes of the 1,281,167 train images.

set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MASTER_PORT=29861

OUT=${CNCLIP_MID_OUT:-/data3/yxfeng/experiments/cnclip_midtrain_l2_imagenet_zh}
INIT=${CNCLIP_MID_INIT:-/data3/yxfeng/experiments/cnclip_midtrain_l2_loop1loss/latest.pt}
DATA=/data3/yxfeng/datasets/imagenet/train
LABELS=/data3/yxfeng/datasets/imagenet/label_cn.txt
mkdir -p "$OUT"
TORCHRUN=/home/yxfeng/miniconda3/envs/www27/bin/torchrun
CLIP=/data3/yxfeng/models/clip/chinese-clip-vit-base-patch16

run_train() {
    local batch="$1"
    local steps="$2"
    local extra=()
    if [[ -f "$OUT/latest.pt" ]]; then
        extra=(--resume "$OUT/latest.pt")
    else
        extra=(--init "$INIT")
    fi
    "$TORCHRUN" --nproc_per_node=8 --master_port="$MASTER_PORT" -m src.mid_training \
        --clip "$CLIP" \
        --data "$DATA" \
        --data-format imagenet-zh \
        --label-file "$LABELS" \
        --output "$OUT" \
        --steps "$steps" \
        --warmup-steps 0 \
        --lr-warmup 400 \
        --batch-size "$batch" \
        --lr 5e-6 \
        --weight-decay 0.01 \
        --num-loops 2 \
        --max-length 64 \
        --num-workers 4 \
        --log-every 20 \
        --save-every 2000 \
        "${extra[@]}"
}

if ! run_train 32 10010; then
    if grep -qi "out of memory" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"; then
        echo "batch 32 ran out of memory, retrying at batch 16" >&2
        run_train 16 20020
    else
        exit 1
    fi
fi
