#!/bin/bash
# GossipCop after the Weibo lr sweep.
# Picks the lr whose mean test macro F1 is highest on Weibo bsz=32
# (the 2e-5 runs plus logs/weibo_lr_sweep.log), then trains LoopFuse and
# the same-backbone baselines. Two GPUs, microbatch 32, accumulation 2,
# so the global batch stays 128. English CLIP max length is 77.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/gossipcop

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-2}"
SUMMARY=logs/gossipcop.log

read -r LR MIN_LR < <("${PY}" - <<'PY'
import re
from collections import defaultdict
from pathlib import Path

root = Path("/home/yxfeng/cscwd27_loopfuse/logs")
groups = defaultdict(list)

def macro_f1(path: Path):
    text = path.read_text(errors="replace").replace("\r", "\n")
    found = re.findall(r"^test accuracy=.*macro_f1=([0-9.]+)", text, re.M)
    return float(found[-1]) if found else None

for path in sorted(root.glob("weibo_bsz32_seed*_m32.log")):
    score = macro_f1(path)
    if score is not None:
        groups["2e-5"].append(score)

sweep = root / "weibo_lr_sweep.log"
if sweep.is_file():
    for line in sweep.read_text(errors="replace").splitlines():
        match = re.search(
            r"OK (lr\S+) seed=(\d+) micro=(\d+) accum=(\d+)", line
        )
        if not match:
            continue
        tag, seed, micro, _accum = match.groups()
        lr = tag[2:]
        path = root / f"weibo_{tag}_seed{seed}_m{micro}.log"
        score = macro_f1(path) if path.is_file() else None
        if score is not None:
            groups[lr].append(score)

if not groups:
    print("2e-5 2e-6")
    raise SystemExit(0)

best_lr, scores = max(groups.items(), key=lambda item: sum(item[1]) / len(item[1]))
min_lr = f"{float(best_lr) / 10:.0e}".replace("e-0", "e-")
print(best_lr, min_lr)
print(
    "candidates "
    + " ".join(
        f"{lr}:{sum(v)/len(v):.4f}/n={len(v)}" for lr, v in sorted(groups.items())
    ),
    file=__import__("sys").stderr,
)
PY
)

echo "START $(date) nproc=${NPROC} host=$(hostname) lr=${LR} min_lr=${MIN_LR}" | tee "$SUMMARY"

run_attempt() {
  local name="$1" model_type="$2" seed="$3" micro="$4" accum="$5" log="$6"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29661 \
    trainer.py --config=gossipcop \
    model_type="${model_type}" \
    output_dir="outputs/gossipcop/${name}/seed_${seed}" \
    run_name="gossipcop-${name}-seed${seed}" \
    num_loops=3 epochs=50 patience=5 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    lr="${LR}" min_lr="${MIN_LR}" \
    seed="${seed}" >> "${log}" 2>&1
}

fail=0
# name model_type. LoopFuse first, then the same-backbone controls.
methods=(
  "loopfuse clip"
  "clip_head clip_head"
  "layer_trajectory layer_trajectory"
  "clip_concat clip_concat"
  "clip_interact clip_interact"
  "stacked stacked"
)
for spec in "${methods[@]}"; do
  read -r name model_type <<< "${spec}"
  for seed in 0 1 2 3; do
    micro=32
    accum=2
    while true; do
      log="logs/gossipcop_${name}_seed${seed}_m${micro}.log"
      echo "===== ${name} seed=${seed} micro=${micro} accum=${accum} lr=${LR} $(date) =====" | tee -a "$SUMMARY"
      rm -rf "outputs/gossipcop/${name}/seed_${seed}"
      status=0
      run_attempt "${name}" "${model_type}" "${seed}" "${micro}" "${accum}" "${log}" || status=$?
      if [ "${status}" -eq 0 ]; then
        echo "OK ${name} seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
        break
      fi
      if grep -Eq "out of memory|OutOfMemoryError" "${log}" && [ "${micro}" -gt 1 ]; then
        micro=$((micro / 2))
        accum=$((accum * 2))
        echo "OOM retry ${name} seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
        continue
      fi
      echo "FAIL ${name} seed=${seed} exit=${status}" | tee -a "$SUMMARY"
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
for path in sorted(root.glob("gossipcop_*_seed*_m*.log")):
    text = path.read_text(errors="replace").replace("\r", "\n")
    found = re.findall(r"^test accuracy=([0-9.]+).*macro_f1=([0-9.]+)", text, re.M)
    if not found:
        continue
    name = path.name.split("_seed")[0].removeprefix("gossipcop_")
    acc, macro = found[-1]
    groups[name].append((float(acc), float(macro)))
print("GossipCop mean over finished seeds")
for name, rows in groups.items():
    acc = sum(item[0] for item in rows) / len(rows)
    macro = sum(item[1] for item in rows) / len(rows)
    print(f"{name} n={len(rows)} acc={acc:.4f} macro_f1={macro:.4f}")
PY

echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
