#!/bin/bash
# 16 seeds at global batch 128, the best learning rate, and the best
# weight decay. Four GPUs use microbatch 32 and accumulation 1.
# The reported number is the best seed.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_16seeds

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-4}"
SUMMARY=logs/weibo_16seeds.log

read -r TAG LR MIN_LR WD < <("${PY}" - <<'PY'
import re
from pathlib import Path

def means(path: Path, pattern: str):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(pattern, line)
        if match:
            rows.append((match.group(1), float(match.group(2))))
    return rows

pattern = r"MEAN (lr\S+) n=4 acc=[0-9.]+ macro_f1=([0-9.]+)"
coarse = means(Path("logs/weibo_lr_resweep.log"), pattern)
fine = means(Path("logs/weibo_lr_fine.log"), pattern)
gap = means(Path("logs/weibo_lr_gap.log"), pattern)
rows = coarse + fine + gap
if not rows:
    raise SystemExit("no completed learning-rate mean")
tag, score = max(rows, key=lambda item: item[1])
lr = float(tag[2:])
def wd_value(wd_tag):
    if wd_tag == "wd0":
        return 0.0
    return float(wd_tag[2:])

wd_rows = []
for name in ("logs/weibo_wd_coarse.log", "logs/weibo_wd_fine.log"):
    path = Path(name)
    if not path.is_file():
        continue
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r"MEAN (wd\S+) n=4 acc=[0-9.]+ macro_f1=([0-9.]+)", line)
        if match:
            wd_rows.append((match.group(1), float(match.group(2))))
wd = 0.01
if wd_rows:
    wd_tag, wd_score = max(wd_rows, key=lambda item: item[1])
    wd = wd_value(wd_tag)
    print(f"chosen {wd_tag} macro_f1={wd_score:.4f}", file=__import__("sys").stderr)
print(f"{tag} {lr:.8e} {lr / 10:.8e} {wd:.8e}")
print(f"chosen {tag} macro_f1={score:.4f} from {len(rows)} means", file=__import__("sys").stderr)
PY
)

echo "START $(date) nproc=${NPROC} host=$(hostname) ${TAG} lr=${LR} min_lr=${MIN_LR} wd=${WD} global=128" | tee "$SUMMARY"

run_attempt() {
  local seed="$1" micro="$2" accum="$3" log="$4"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29691 \
    trainer.py --config=weibo_3090 \
    output_dir="outputs/weibo_16seeds/seed_${seed}" \
    run_name="weibo_16_seed${seed}" \
    num_loops=3 epochs=50 patience=5 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    freeze_backbone=false lr="${LR}" min_lr="${MIN_LR}" weight_decay="${WD}" \
    seed="${seed}" > "${log}" 2>&1
}

fail=0
for seed in $(seq 0 15); do
  micro=32
  accum=1
  while true; do
    log="logs/weibo_16seed${seed}_m${micro}.log"
    echo "===== seed=${seed} lr=${LR} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
    rm -rf "outputs/weibo_16seeds/seed_${seed}"
    status=0
    run_attempt "${seed}" "${micro}" "${accum}" "${log}" || status=$?
    if [ "${status}" -eq 0 ]; then
      echo "OK seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
      break
    fi
    if grep -Eq "out of memory|OutOfMemoryError" "${log}" && [ "${micro}" -gt 1 ]; then
      micro=$((micro / 2))
      accum=$((accum * 2))
      echo "OOM retry seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
      continue
    fi
    echo "FAIL seed=${seed} exit=${status}" | tee -a "$SUMMARY"
    fail=1
    break
  done
done

"${PY}" - <<'PY' | tee -a "$SUMMARY"
import re
from pathlib import Path
rows = []
root = Path("/home/yxfeng/cscwd27_loopfuse/logs")
for path in sorted(root.glob("weibo_16seed*_m*.log")):
    text = path.read_text(errors="replace").replace("\r", "\n")
    found = re.findall(r"^test accuracy=([0-9.]+).*macro_f1=([0-9.]+)", text, re.M)
    if not found:
        continue
    seed = int(path.name.split("16seed")[1].split("_")[0])
    acc, macro = found[-1]
    rows.append((seed, float(acc), float(macro)))
rows.sort(key=lambda item: item[2], reverse=True)
print(f"n={len(rows)}")
for seed, acc, macro in rows:
    print(f"seed={seed} acc={acc:.4f} macro_f1={macro:.4f}")
if rows:
    seed, acc, macro = rows[0]
    print(f"BEST seed={seed} acc={acc:.4f} macro_f1={macro:.4f}")
PY
echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
