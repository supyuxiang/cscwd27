#!/bin/bash
# Fine weight-decay sweep around the best coarse point.
# Four nearby values, four seeds, same learning rate and global batch 128.
# If the best coarse value is 0, the neighborhood is the small positive
# values that the coarse grid skipped.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_wd_fine

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-4}"
SUMMARY=logs/weibo_wd_fine.log

mapfile -t HEADER < <("${PY}" - <<'PY'
import re
from pathlib import Path

def lr_means():
    pattern = r"MEAN (lr\S+) n=4 acc=[0-9.]+ macro_f1=([0-9.]+)"
    rows = []
    for name in ("logs/weibo_lr_resweep.log", "logs/weibo_lr_fine.log", "logs/weibo_lr_gap.log"):
        path = Path(name)
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            match = re.search(pattern, line)
            if match:
                rows.append((match.group(1), float(match.group(2))))
    return rows

def wd_value(tag):
    if tag == "wd0":
        return 0.0
    return float(tag[2:])

coarse = Path("logs/weibo_wd_coarse.log").read_text(errors="replace")
rows = []
for line in coarse.splitlines():
    match = re.search(r"MEAN (wd\S+) n=4 acc=[0-9.]+ macro_f1=([0-9.]+)", line)
    if match:
        rows.append((match.group(1), float(match.group(2))))
if not rows:
    raise SystemExit("coarse weight-decay sweep has no complete MEAN line")
best_tag, best_score = max(rows, key=lambda item: item[1])
best = wd_value(best_tag)
done = {wd_value(tag) for tag, _score in rows}
lr_rows = lr_means()
lr_tag, lr_score = max(lr_rows, key=lambda item: item[1])
lr = float(lr_tag[2:])
print(f"META {lr_tag} {lr:.8e} {lr / 10:.8e}")
print(f"best coarse {best_tag} macro_f1={best_score:.4f}", file=__import__("sys").stderr)
if best == 0:
    candidates = [1e-5, 3e-5, 3e-4, 3e-3]
else:
    candidates = [best * factor for factor in (0.7, 0.85, 1.2, 1.4)]
for wd in candidates:
    if any(abs(wd - old) <= max(1e-12, 0.02 * max(wd, old)) for old in done):
        continue
    tag = "wd0" if wd == 0 else f"wd{wd:.3e}".replace("+", "").replace("e-0", "e-")
    print(f"{tag} {wd:.8e}")
PY
)

read -r _kind LR_TAG LR MIN_LR <<< "${HEADER[0]}"
SETTINGS=("${HEADER[@]:1}")
if [ "${#SETTINGS[@]}" -eq 0 ]; then
  echo "no new weight-decay values to run" | tee "$SUMMARY"
  exit 0
fi

echo "START $(date) nproc=${NPROC} host=$(hostname) lr=${LR} min_lr=${MIN_LR}" | tee "$SUMMARY"
printf '%s\n' "${SETTINGS[@]}" | tee -a "$SUMMARY"

run_attempt() {
  local tag="$1" wd="$2" seed="$3" micro="$4" accum="$5" log="$6"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29721 \
    trainer.py --config=weibo_3090 \
    output_dir="outputs/weibo_wd_fine/${tag}_seed${seed}" \
    run_name="weibo_wd_fine_${tag}_seed${seed}" \
    num_loops=3 epochs=50 patience=5 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    freeze_backbone=false lr="${LR}" min_lr="${MIN_LR}" weight_decay="${wd}" \
    seed="${seed}" > "${log}" 2>&1
}

fail=0
for spec in "${SETTINGS[@]}"; do
  read -r tag wd <<< "${spec}"
  for seed in 0 1 2 3; do
    micro=32
    accum=1
    while true; do
      log="logs/weibo_wd_fine_${tag}_seed${seed}_m${micro}.log"
      echo "===== ${tag} wd=${wd} lr=${LR} seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
      rm -rf "outputs/weibo_wd_fine/${tag}_seed${seed}"
      status=0
      run_attempt "${tag}" "${wd}" "${seed}" "${micro}" "${accum}" "${log}" || status=$?
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
for path in sorted(root.glob("weibo_wd_fine_wd*_seed*_m*.log")):
    text = path.read_text(errors="replace").replace("\r", "\n")
    found = re.findall(r"^test accuracy=([0-9.]+).*macro_f1=([0-9.]+)", text, re.M)
    if not found:
        continue
    tag = path.name.split("_seed")[0].removeprefix("weibo_wd_fine_")
    acc, macro = found[-1]
    groups[tag].append((float(acc), float(macro)))
print("wd fine means")
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
    print(f"BEST_WD_FINE {best[0]} acc={best[2]:.4f} macro_f1={best[1]:.4f}")
PY
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
