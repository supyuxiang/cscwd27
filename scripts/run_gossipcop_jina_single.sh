#!/bin/bash
# Same jina-clip LoopFuse backbone on GossipCop, one loop only.
# Two 3090s. Learning rate 2e-5. Global batch stays 32.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/gossipcop_jina

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export http_proxy="${http_proxy:-http://192.168.102.101:7890}"
export https_proxy="${https_proxy:-http://192.168.102.101:7890}"
export HF_HUB_DISABLE_XET=1

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-2}"
SUMMARY=logs/gossipcop_jina_single.log
echo "START $(date) nproc=${NPROC} host=$(hostname)" | tee "$SUMMARY"

run_attempt() {
  local seed="$1" micro="$2" accum="$3" log="$4"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29672 \
    trainer.py --config=gossipcop_jina \
    model_type=jina \
    output_dir="outputs/gossipcop_jina/single/seed_${seed}" \
    run_name="gossipcop-jina-single-seed${seed}" \
    num_loops=1 epochs=50 patience=5 \
    max_text_length=2048 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    lr=2e-5 min_lr=2e-6 \
    seed="${seed}" > "${log}" 2>&1
}

fail=0
if [ $((32 % NPROC)) -ne 0 ]; then
  echo "NPROC=${NPROC} does not divide global batch 32" | tee -a "$SUMMARY"
  exit 1
fi
for seed in 0 1 2 3; do
  micro=1
  accum=$((32 / NPROC / micro))
  while true; do
    log="logs/gossipcop_jina_single_seed${seed}_m${micro}.log"
    echo "===== single seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
    rm -rf "outputs/gossipcop_jina/single/seed_${seed}"
    status=0
    run_attempt "${seed}" "${micro}" "${accum}" "${log}" || status=$?
    if [ "${status}" -eq 0 ]; then
      echo "OK single seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
      break
    fi
    if grep -Eq "out of memory|OutOfMemoryError" "${log}" && [ "${micro}" -gt 1 ]; then
      micro=$((micro / 2))
      accum=$((accum * 2))
      echo "OOM retry single seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
      continue
    fi
    echo "FAIL single seed=${seed} exit=${status}" | tee -a "$SUMMARY"
    fail=1
    break
  done
done
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
