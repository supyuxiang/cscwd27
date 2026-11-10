#!/bin/bash
# Weibo LoopFuse, global batch 128 on 2x3090 (microbatch 32, accumulation 2).
# num_loops=3, 50 epochs, patience 5.
# Log-spaced learning rates. min_lr is lr/10.
# Each learning rate runs seeds 0-3. The representative is the mean test
# accuracy and macro F1 of those four seeds.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_lr_resweep

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-2}"
SUMMARY=logs/weibo_lr_resweep.log
echo "START $(date) nproc=${NPROC} host=$(hostname)" | tee "$SUMMARY"

run_attempt() {
  local tag="$1" lr="$2" min_lr="$3" seed="$4" micro="$5" accum="$6" log="$7"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29651 \
    trainer.py --config=weibo_3090 \
    output_dir="outputs/weibo_lr_resweep/${tag}_seed${seed}" \
    run_name="weibo_${tag}_seed${seed}" \
    num_loops=3 epochs=50 patience=5 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    freeze_backbone=false lr="${lr}" min_lr="${min_lr}" \
    seed="${seed}" > "${log}" 2>&1
}

summarize_lr() {
  local tag="$1"
  "${PY}" - "$tag" "$SUMMARY" <<'PY'
import re
import sys
from pathlib import Path

tag, summary = sys.argv[1], Path(sys.argv[2])
root = Path("/home/yxfeng/cscwd27_loopfuse/logs")
rows = []
for seed in range(4):
    scored = []
    for path in sorted(root.glob(f"weibo_{tag}_seed{seed}_m*.log")):
        text = path.read_text(errors="replace").replace("\r", "\n")
        found = re.findall(r"^test accuracy=([0-9.]+).*macro_f1=([0-9.]+)", text, re.M)
        if found:
            scored.append((path, float(found[-1][0]), float(found[-1][1])))
    if scored:
        path, acc, macro = scored[-1]
        rows.append((seed, acc, macro, path.name))
if len(rows) != 4:
    line = f"MEAN {tag} n={len(rows)} incomplete"
else:
    acc = sum(item[1] for item in rows) / 4
    macro = sum(item[2] for item in rows) / 4
    detail = " ".join(f"s{seed}={macro:.4f}" for seed, _acc, macro, _name in rows)
    line = f"MEAN {tag} n=4 acc={acc:.4f} macro_f1={macro:.4f} {detail}"
print(line)
with summary.open("a", encoding="utf-8") as handle:
    handle.write(line + "\n")
PY
}

fail=0
# tag lr min_lr
settings=(
  "lr1e-6 1e-6 1e-7"
  "lr2e-6 2e-6 2e-7"
  "lr5e-6 5e-6 5e-7"
  "lr1e-5 1e-5 1e-6"
  "lr2e-5 2e-5 2e-6"
  "lr5e-5 5e-5 5e-6"
  "lr1e-4 1e-4 1e-5"
  "lr2e-4 2e-4 2e-5"
)
for spec in "${settings[@]}"; do
  read -r tag lr min_lr <<< "${spec}"
  for seed in 0 1 2 3; do
    micro=32
    accum=2
    while true; do
      log="logs/weibo_${tag}_seed${seed}_m${micro}.log"
      echo "===== ${tag} lr=${lr} min_lr=${min_lr} seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
      rm -rf "outputs/weibo_lr_resweep/${tag}_seed${seed}"
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
  summarize_lr "${tag}"
done
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
