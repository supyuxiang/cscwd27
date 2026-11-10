#!/bin/bash
# Weibo LoopFuse, num_loops=3, 50 epochs, epoch-level early stop.
# Sweep per-GPU batch 8, 16, 32, 64. Each setting uses 4 seeds.
# If a microbatch OOMs, halve it and double grad accumulation so the
# per-GPU effective batch stays the same.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_bsz_sweep

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-4}"
SUMMARY=logs/weibo_bsz_sweep.log
echo "START $(date) nproc=${NPROC} host=$(hostname)" | tee "$SUMMARY"

run_attempt() {
  local bsz="$1" seed="$2" micro="$3" accum="$4" log="$5"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29640 \
    trainer.py --config=weibo_3090 \
    output_dir="outputs/weibo_bsz_sweep/bsz${bsz}_seed${seed}" \
    run_name="weibo_bsz${bsz}_seed${seed}" \
    num_loops=3 epochs=50 patience=5 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    freeze_backbone=false lr=2e-5 min_lr=2e-6 \
    seed="${seed}" >> "${log}" 2>&1
}

fail=0
for bsz in 8 16 32 64; do
  for seed in 0 1 2 3; do
    micro="${bsz}"
    accum=1
    while true; do
      log="logs/weibo_bsz${bsz}_seed${seed}_m${micro}.log"
      echo "===== bsz=${bsz} seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
      rm -rf "outputs/weibo_bsz_sweep/bsz${bsz}_seed${seed}"
      status=0
      run_attempt "${bsz}" "${seed}" "${micro}" "${accum}" "${log}" || status=$?
      if [ "${status}" -eq 0 ]; then
        echo "OK bsz=${bsz} seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
        break
      fi
      if grep -Eq "out of memory|OutOfMemoryError" "${log}" && [ "${micro}" -gt 1 ]; then
        micro=$((micro / 2))
        accum=$((accum * 2))
        echo "OOM retry bsz=${bsz} seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
        continue
      fi
      echo "FAIL bsz=${bsz} seed=${seed} exit=${status}" | tee -a "$SUMMARY"
      fail=1
      break
    done
  done
done
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
