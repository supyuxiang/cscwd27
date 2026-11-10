#!/bin/bash
# Same Weibo setup as configs/weibo_3090.yaml: 1 epoch, lr 2e-5, Chinese CLIP.
# Run on the existing 2x 3090 allocation:
#   srun --jobid=<job> --overlap --ntasks=1 --gres=gpu:2 bash scripts/run_weibo_baselines.sh
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
SUMMARY=logs/weibo_baselines.log

run() {
  local name="$1"
  shift
  local log="logs/weibo_${name}.log"
  echo "===== ${name} $* $(date) =====" | tee -a "$SUMMARY" >> "$log"
  local status=0
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:29531" \
    --rdzv_id="weibo-${name}" \
    trainer.py --config=weibo_3090 \
    model_type="${name}" \
    output_dir="outputs/weibo_${name}" \
    run_name="weibo-${name}" \
    "$@" >> "$log" 2>&1 || status=$?
  echo "===== ${name} exit ${status} $(date) =====" | tee -a "$SUMMARY" >> "$log"
  return "${status}"
}

echo "START $(date) host=$(hostname) gpus=$(nvidia-smi -L | wc -l)" | tee "$SUMMARY"

fail=0
for name in clip_concat clip_interact clip_head layer_trajectory relaxed; do
  run "${name}" || fail=1
done

if ! run stacked batch_size=4 grad_accum_steps=2; then
  echo "stacked retry batch_size=2" | tee -a "$SUMMARY"
  run stacked batch_size=2 grad_accum_steps=4 || fail=1
fi

echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
