#!/bin/bash
# LoopFuse full finetune, 4x RTX 4090 on node4 (half of the node's 96 CPUs).
# Submit from anywhere:
#   sbatch scripts/train_node4.sbatch
#   sbatch scripts/train_node4.sbatch configs/twitter.yaml
#SBATCH --job-name=loopfuse
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodelist=node4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT=/home/yxfeng/cscwd27_loopfuse
cd "$ROOT"
mkdir -p logs

CONFIG="${1:-configs/weibo.yaml}"
PY=/home/yxfeng/miniconda3/envs/www27/bin/python

# 4090 has no NVLink; PCIe P2P on this cluster hangs NCCL if left on.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "job=${SLURM_JOB_ID} node=$(hostname) config=${CONFIG}"
"${PY}" -c "import torch; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())"

"${PY}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
  --rdzv_id="${SLURM_JOB_ID}" \
  train_ddp.py \
  --config "${CONFIG}"
