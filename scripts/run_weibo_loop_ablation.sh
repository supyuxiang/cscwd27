#!/bin/bash
# Weibo loop ablation: num_loops = 1..16, everything else from weibo_3090.yaml.
#   srun --jobid=<job> --overlap --ntasks=1 --gres=gpu:2 bash scripts/run_weibo_loop_ablation.sh
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_loops_ablation

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
SUMMARY=logs/weibo_loop_ablation.log

run() {
  local loops="$1"
  local tag
  tag="$(printf 'loop_%02d' "${loops}")"
  local log="logs/weibo_${tag}.log"
  echo "===== ${tag} $(date) =====" | tee -a "$SUMMARY" >> "$log"
  local status=0
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:29541" \
    --rdzv_id="weibo-${tag}" \
    trainer.py --config=weibo_3090 \
    model_type=chinese \
    num_loops="${loops}" \
    output_dir="outputs/weibo_loops_ablation/${tag}" \
    run_name="weibo-${tag}" \
    >> "$log" 2>&1 || status=$?
  echo "===== ${tag} exit ${status} $(date) =====" | tee -a "$SUMMARY" >> "$log"
  return "${status}"
}

echo "START $(date) host=$(hostname)" | tee "$SUMMARY"
fail=0
for loops in $(seq 1 16); do
  run "${loops}" || fail=1
done
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
