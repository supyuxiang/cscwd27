#!/bin/bash
# Weibo LoopFuse at the bsz=32 setting from the 4x4090 batch sweep.
# Two 3090s, microbatch 32, grad accumulation 2: global batch stays 128.
# num_loops=3, 50 epochs, patience 5.
# Sweep learning rate. min_lr stays at lr/10, matching the 2e-5 / 2e-6 run.
# The 2e-5 point is already in outputs/weibo_bsz_sweep and is not repeated.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_lr_sweep

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-2}"
SUMMARY=logs/weibo_lr_sweep.log
echo "START $(date) nproc=${NPROC} host=$(hostname)" | tee "$SUMMARY"

run_attempt() {
  local tag="$1" lr="$2" min_lr="$3" seed="$4" micro="$5" accum="$6" log="$7"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29651 \
    trainer.py --config=weibo_3090 \
    output_dir="outputs/weibo_lr_sweep/${tag}_seed${seed}" \
    run_name="weibo_${tag}_seed${seed}" \
    num_loops=3 epochs=50 patience=5 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    freeze_backbone=false lr="${lr}" min_lr="${min_lr}" \
    seed="${seed}" >> "${log}" 2>&1
}

fail=0
# tag lr min_lr
settings=(
  "lr5e-6 5e-6 5e-7"
  "lr1e-5 1e-5 1e-6"
  "lr5e-5 5e-5 5e-6"
  "lr1e-4 1e-4 1e-5"
)
for spec in "${settings[@]}"; do
  read -r tag lr min_lr <<< "${spec}"
  for seed in 0 1 2 3; do
    micro=32
    accum=2
    while true; do
      log="logs/weibo_${tag}_seed${seed}_m${micro}.log"
      echo "===== ${tag} lr=${lr} min_lr=${min_lr} seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
      rm -rf "outputs/weibo_lr_sweep/${tag}_seed${seed}"
      status=0
      run_attempt "${tag}" "${lr}" "${min_lr}" "${seed}" "${micro}" "${accum}" "${log}" || status=$?
      if [ "${status}" -eq 0 ]; then
        echo "OK ${tag} seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
        break
      fi
      if grep -Eq "out of memory|OutOfMemoryError" "${log}" && [ "${micro}" -gt 1 ]; then
        micro=$((micro / 2))
        accum=$((accum * 2))
        echo "OOM retry ${tag} seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
        continue
      fi
      echo "FAIL ${tag} seed=${seed} exit=${status}" | tee -a "$SUMMARY"
      fail=1
      break
    done
  done
done
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
