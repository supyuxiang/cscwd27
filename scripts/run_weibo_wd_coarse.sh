#!/bin/bash
# Coarse weight-decay sweep at the best learning rate.
# The learning rate is the highest 4-seed mean test macro F1 among the
# coarse, fine, and gap sweeps. Weight decay is log-spaced around 0.01.
# 4 GPUs, microbatch 32, accumulation 1, global batch 128. Four seeds each.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_wd_coarse

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-4}"
SUMMARY=logs/weibo_wd_coarse.log

read -r LR_TAG LR MIN_LR < <("${PY}" - <<'PY'
import re
from pathlib import Path

def means(path, pattern):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(pattern, line)
        if match:
            rows.append((match.group(1), float(match.group(2))))
    return rows

pattern = r"MEAN (lr\S+) n=4 acc=[0-9.]+ macro_f1=([0-9.]+)"
rows = []
rows += means(Path("logs/weibo_lr_resweep.log"), pattern)
rows += means(Path("logs/weibo_lr_fine.log"), pattern)
rows += means(Path("logs/weibo_lr_gap.log"), pattern)
if not rows:
    raise SystemExit("no completed learning-rate mean")
tag, score = max(rows, key=lambda item: item[1])
lr = float(tag[2:])
print(f"{tag} {lr:.8e} {lr / 10:.8e}")
print(f"chosen {tag} macro_f1={score:.4f} from {len(rows)} means", file=__import__("sys").stderr)
PY
)

echo "START $(date) nproc=${NPROC} host=$(hostname) lr=${LR} min_lr=${MIN_LR} from ${LR_TAG}" | tee "$SUMMARY"

run_attempt() {
  local tag="$1" wd="$2" seed="$3" micro="$4" accum="$5" log="$6"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29711 \
    trainer.py --config=weibo_3090 \
    output_dir="outputs/weibo_wd_coarse/${tag}_seed${seed}" \
    run_name="weibo_wd_${tag}_seed${seed}" \
    num_loops=3 epochs=50 patience=5 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    freeze_backbone=false lr="${LR}" min_lr="${MIN_LR}" weight_decay="${wd}" \
    seed="${seed}" > "${log}" 2>&1
}

fail=0
# tag weight_decay. 0.01 is the value used in the learning-rate sweeps.
settings=(
  "wd0 0"
  "wd1e-4 1e-4"
  "wd3e-4 3e-4"
  "wd1e-3 1e-3"
  "wd3e-3 3e-3"
  "wd1e-2 1e-2"
  "wd3e-2 3e-2"
  "wd1e-1 1e-1"
)
for spec in "${settings[@]}"; do
  read -r tag wd <<< "${spec}"
  for seed in 0 1 2 3; do
    micro=32
    accum=1
    while true; do
      log="logs/weibo_wd_${tag}_seed${seed}_m${micro}.log"
      echo "===== ${tag} wd=${wd} lr=${LR} seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
      rm -rf "outputs/weibo_wd_coarse/${tag}_seed${seed}"
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
for path in sorted(root.glob("weibo_wd_wd*_seed*_m*.log")):
    text = path.read_text(errors="replace").replace("\r", "\n")
    found = re.findall(r"^test accuracy=([0-9.]+).*macro_f1=([0-9.]+)", text, re.M)
    if not found:
        continue
    tag = path.name.split("_seed")[0].removeprefix("weibo_wd_")
    acc, macro = found[-1]
    groups[tag].append((float(acc), float(macro)))
print("wd coarse means")
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
    print(f"BEST_WD {best[0]} acc={best[2]:.4f} macro_f1={best[1]:.4f}")
PY
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
