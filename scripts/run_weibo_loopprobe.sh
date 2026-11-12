#!/bin/bash
# LoopProbe on Weibo: loopback reentry, classify the last loop only.
# Loops 1, 3, 5 and 8, eight seeds split across two nodes.
# global batch 128, backbone lr 2e-5, new modules 2e-4, weight decay 1e-4.
# 4 GPUs, microbatch 32, accumulation 1. An OOM halves the microbatch
# and doubles accumulation so the global batch stays 128.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_loopprobe

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-4}"
SPLIT="${SPLIT:-all}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
SUMMARY="logs/weibo_loopprobe_${SPLIT}.log"
OUT=outputs/weibo_loopprobe
LR=2e-5
MIN_LR=2e-6
WD=1e-4

echo "START $(date) split=${SPLIT} seeds=${SEEDS} nproc=${NPROC} host=$(hostname) lr=${LR} min_lr=${MIN_LR} wd=${WD} batch=128 model=loopprobe" | tee "$SUMMARY"

run_attempt() {
  local loops="$1" seed="$2" micro="$3" accum="$4" log="$5"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29771 \
    trainer.py --config=weibo_3090 \
    output_dir="${OUT}/loop${loops}_seed${seed}" \
    run_name="weibo_loopprobe_loop${loops}_seed${seed}" \
    num_loops="${loops}" epochs=50 patience=5 max_text_length=512 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    model_type=loopprobe freeze_backbone=false lr="${LR}" min_lr="${MIN_LR}" weight_decay="${WD}" \
    seed="${seed}" > "${log}" 2>&1
}

seed_done() {
  "${PY}" - "$1" "$2" "$OUT/results.jsonl" <<'PY'
import json, sys
from pathlib import Path
loop, seed, path = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
if not path.is_file():
    raise SystemExit(1)
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    if int(item["loop"]) == loop and int(item["seed"]) == seed:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

record_seed() {
  local loops="$1" seed="$2" log="$3"
  flock "$OUT/record.lock" "${PY}" - "$loops" "$seed" "$log" "$OUT" <<'PY'
import json, re, sys
from pathlib import Path
loops, seed, log, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
text = log.read_text(errors="replace").replace("\r", "\n")
found = re.findall(r"^test accuracy=([0-9.]+).*macro_f1=([0-9.]+)", text, re.M)
epochs = re.findall(r"epoch (\d+)/50", text)
if not found:
    raise SystemExit(f"no test line in {log}")
acc, macro = float(found[-1][0]), float(found[-1][1])
row = {
    "loop": loops,
    "seed": seed,
    "test_acc": acc,
    "macro_f1": macro,
    "epochs_ran": int(epochs[-1]) if epochs else None,
}
out.mkdir(parents=True, exist_ok=True)
with (out / "results.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
rows = []
for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        rows.append(json.loads(line))
latest = {}
for item in rows:
    latest[(item["loop"], item["seed"])] = item
rows = list(latest.values())
csv = ["loop,seed,test_acc,macro_f1,epochs_ran"]
for item in sorted(rows, key=lambda item: (item["loop"], item["seed"])):
    csv.append(f"{item['loop']},{item['seed']},{item['test_acc']:.4f},{item['macro_f1']:.4f},{item['epochs_ran']}")
(out / "seeds.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
best_rows = ["loop,best_seed,best_acc,best_macro_f1,n_seeds"]
by_loop = {}
for item in rows:
    by_loop.setdefault(item["loop"], []).append(item)
summary = []
for loop_id in sorted(by_loop):
    group = by_loop[loop_id]
    best = max(group, key=lambda item: item["test_acc"])
    summary.append({
        "loop": loop_id,
        "n_seeds": len(group),
        "best_seed": best["seed"],
        "best_acc": best["test_acc"],
        "best_macro_f1": best["macro_f1"],
        "seeds": sorted(group, key=lambda item: item["seed"]),
    })
    best_rows.append(
        f"{loop_id},{best['seed']},{best['test_acc']:.4f},{best['macro_f1']:.4f},{len(group)}"
    )
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out / "best_by_loop.csv").write_text("\n".join(best_rows) + "\n", encoding="utf-8")
print(f"RECORDED loop={loops} seed={seed} acc={acc:.4f} macro_f1={macro:.4f}")
PY
}

fail=0
for loops in 1 3 5 8; do
  for seed in ${SEEDS}; do
    if seed_done "${loops}" "${seed}"; then
      echo "SKIP loop=${loops} seed=${seed} already recorded" | tee -a "$SUMMARY"
      continue
    fi
    micro=32
    accum=1
    while true; do
      log="logs/weibo_loopprobe_loop${loops}_seed${seed}_m${micro}.log"
      echo "===== loop=${loops} seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
      rm -rf "${OUT}/loop${loops}_seed${seed}"
      status=0
      run_attempt "${loops}" "${seed}" "${micro}" "${accum}" "${log}" || status=$?
      if [ "${status}" -eq 0 ]; then
        echo "OK loop=${loops} seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
        record_seed "${loops}" "${seed}" "${log}" | tee -a "$SUMMARY"
        break
      fi
      if grep -Eq "out of memory|OutOfMemoryError" "${log}" && [ "${micro}" -gt 1 ]; then
        micro=$((micro / 2))
        accum=$((accum * 2))
        echo "OOM retry loop=${loops} seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
        continue
      fi
      echo "FAIL loop=${loops} seed=${seed} exit=${status}" | tee -a "$SUMMARY"
      exit 1
    done
  done
  flock "$OUT/record.lock" "${PY}" - "$loops" "$OUT" <<'PY' | tee -a "$SUMMARY"
import json, sys
from pathlib import Path
loop_id = int(sys.argv[1])
path = Path(sys.argv[2]) / "summary.json"
if not path.is_file():
    raise SystemExit("no summary yet")
summary = json.loads(path.read_text(encoding="utf-8"))
match = next((item for item in summary if item["loop"] == loop_id), None)
if match is None:
    raise SystemExit(f"loop {loop_id} has no recorded seed")
print(
    f"BEST loop={match['loop']} seed={match['best_seed']} "
    f"n={match['n_seeds']} acc={match['best_acc']:.4f} macro_f1={match['best_macro_f1']:.4f}"
)
PY
done

echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
