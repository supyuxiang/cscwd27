#!/bin/bash
# Learning rates in the 1e-5 to 5e-5 band that neither the coarse grid
# nor the fine grid covered. Four seeds each. 4 GPUs, global batch 128.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_lr_gap

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-4}"
SUMMARY=logs/weibo_lr_gap.log
echo "START $(date) nproc=${NPROC} host=$(hostname)" | tee "$SUMMARY"

run_attempt() {
  local tag="$1" lr="$2" min_lr="$3" seed="$4" micro="$5" accum="$6" log="$7"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29701 \
    trainer.py --config=weibo_3090 \
    output_dir="outputs/weibo_lr_gap/${tag}_seed${seed}" \
    run_name="weibo_gap_${tag}_seed${seed}" \
    num_loops=3 epochs=50 patience=5 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    freeze_backbone=false lr="${lr}" min_lr="${min_lr}" \
    seed="${seed}" > "${log}" 2>&1
}

fail=0
settings=(
  "lr1.2e-5 1.2e-5 1.2e-6"
  "lr2.2e-5 2.2e-5 2.2e-6"
  "lr3.5e-5 3.5e-5 3.5e-6"
  "lr4e-5 4e-5 4e-6"
)
for spec in "${settings[@]}"; do
  read -r tag lr min_lr <<< "${spec}"
  for seed in 0 1 2 3; do
    micro=32
    accum=1
    while true; do
      log="logs/weibo_gap_${tag}_seed${seed}_m${micro}.log"
      echo "===== ${tag} lr=${lr} min_lr=${min_lr} seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
      rm -rf "outputs/weibo_lr_gap/${tag}_seed${seed}"
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

"${PY}" - <<'PY' | tee -a "$SUMMARY"
import re
from collections import defaultdict
from pathlib import Path
root = Path("/home/yxfeng/cscwd27_loopfuse/logs")
groups = defaultdict(list)
for path in sorted(root.glob("weibo_gap_lr*_seed*_m*.log")):
    text = path.read_text(errors="replace").replace("\r", "\n")
    found = re.findall(r"^test accuracy=([0-9.]+).*macro_f1=([0-9.]+)", text, re.M)
    if not found:
        continue
    tag = path.name.split("_seed")[0].removeprefix("weibo_gap_")
    acc, macro = found[-1]
    groups[tag].append((float(acc), float(macro)))
print("gap means")
for tag, rows in groups.items():
    if len(rows) != 4:
        print(f"MEAN {tag} n={len(rows)} incomplete")
        continue
    acc = sum(item[0] for item in rows) / 4
    macro = sum(item[1] for item in rows) / 4
    print(f"MEAN {tag} n=4 acc={acc:.4f} macro_f1={macro:.4f}")
PY
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
