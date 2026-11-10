#!/bin/bash
# Fine learning-rate sweep after logs/weibo_lr_resweep.log.
# Reads the coarse MEAN lines, takes the lr with the highest mean test
# macro F1, and runs four nearby rates at seeds 0-3 on the same 2x3090
# global-batch-128 recipe.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_lr_fine

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-2}"
SUMMARY=logs/weibo_lr_fine.log
COARSE=logs/weibo_lr_resweep.log

mapfile -t SETTINGS < <("${PY}" - "$COARSE" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="replace")
rows = []
for line in text.splitlines():
    match = re.search(r"MEAN (lr\S+) n=4 acc=[0-9.]+ macro_f1=([0-9.]+)", line)
    if match:
        rows.append((match.group(1), float(match.group(2))))
if not rows:
    raise SystemExit("coarse sweep has no complete MEAN line")
best_tag, best_score = max(rows, key=lambda item: item[1])
best = float(best_tag[2:])
# Neighbors on a log scale. The coarse point itself is not repeated.
factors = (0.7, 0.85, 1.2, 1.4)
print(f"best {best_tag} macro_f1={best_score:.4f}", file=sys.stderr)
for factor in factors:
    lr = best * factor
    tag = f"lr{lr:.3e}".replace("+", "").replace("e-0", "e-")
    min_lr = lr / 10
    print(f"{tag} {lr:.8e} {min_lr:.8e}")
PY
)

echo "START $(date) nproc=${NPROC} host=$(hostname)" | tee "$SUMMARY"
printf '%s\n' "${SETTINGS[@]}" | tee -a "$SUMMARY"

run_attempt() {
  local tag="$1" lr="$2" min_lr="$3" seed="$4" micro="$5" accum="$6" log="$7"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29681 \
    trainer.py --config=weibo_3090 \
    output_dir="outputs/weibo_lr_fine/${tag}_seed${seed}" \
    run_name="weibo_fine_${tag}_seed${seed}" \
    num_loops=3 epochs=50 patience=5 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    freeze_backbone=false lr="${lr}" min_lr="${min_lr}" \
    seed="${seed}" > "${log}" 2>&1
}

fail=0
for spec in "${SETTINGS[@]}"; do
  read -r tag lr min_lr <<< "${spec}"
  for seed in 0 1 2 3; do
    micro=32
    accum=2
    while true; do
      log="logs/weibo_fine_${tag}_seed${seed}_m${micro}.log"
      echo "===== ${tag} lr=${lr} min_lr=${min_lr} seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
      rm -rf "outputs/weibo_lr_fine/${tag}_seed${seed}"
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
for path in sorted(root.glob("weibo_fine_lr*_seed*_m*.log")):
    text = path.read_text(errors="replace").replace("\r", "\n")
    found = re.findall(r"^test accuracy=([0-9.]+).*macro_f1=([0-9.]+)", text, re.M)
    if not found:
        continue
    tag = path.name.split("_seed")[0].removeprefix("weibo_fine_")
    acc, macro = found[-1]
    groups[tag].append((float(acc), float(macro)))
print("fine means")
best = None
for tag, rows in groups.items():
    if len(rows) != 4:
        print(f"MEAN {tag} n={len(rows)} incomplete")
        continue
    acc = sum(item[0] for item in rows) / 4
    macro = sum(item[1] for item in rows) / 4
    print(f"MEAN {tag} n=4 acc={acc:.4f} macro_f1={macro:.4f}")
    if best is None or macro > best[1]:
        best = (tag, macro, acc)
if best:
    print(f"BEST_FINE {best[0]} acc={best[2]:.4f} macro_f1={best[1]:.4f}")
PY
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
