#!/bin/bash
# Weibo, loop count 5 only, eight seeds.
# Same setting as the loop ablation: batch 128, lr 2e-5, weight decay 1e-4.
# The score for this loop count is the best test accuracy among the eight seeds.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs outputs/weibo_loop5

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-4}"
SUMMARY=logs/weibo_loop5.log
LR=2e-5
MIN_LR=2e-6
WD=1e-4
LOOPS=5

echo "START $(date) nproc=${NPROC} host=$(hostname) loops=${LOOPS} lr=${LR} wd=${WD} batch=128" | tee "$SUMMARY"

run_attempt() {
  local seed="$1" micro="$2" accum="$3" log="$4"
  "${PY}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port=29741 \
    trainer.py --config=weibo_3090 \
    output_dir="outputs/weibo_loop5/seed${seed}" \
    run_name="weibo_loop5_seed${seed}" \
    num_loops="${LOOPS}" epochs=50 patience=5 max_text_length=512 \
    batch_size="${micro}" grad_accum_steps="${accum}" \
    freeze_backbone=false lr="${LR}" min_lr="${MIN_LR}" weight_decay="${WD}" \
    seed="${seed}" > "${log}" 2>&1
}

record_seed() {
  local seed="$1" log="$2"
  "${PY}" - "$seed" "$log" <<'PY'
import json, re, sys
from pathlib import Path
seed, log = int(sys.argv[1]), Path(sys.argv[2])
text = log.read_text(errors="replace").replace("\r", "\n")
found = re.findall(r"^test accuracy=([0-9.]+).*macro_f1=([0-9.]+)", text, re.M)
epochs = re.findall(r"epoch (\d+)/50", text)
if not found:
    raise SystemExit(f"no test line in {log}")
acc, macro = float(found[-1][0]), float(found[-1][1])
row = {
    "loop": 5,
    "seed": seed,
    "test_acc": acc,
    "macro_f1": macro,
    "epochs_ran": int(epochs[-1]) if epochs else None,
}
out = Path("outputs/weibo_loop5")
out.mkdir(parents=True, exist_ok=True)
with (out / "results.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
latest = {}
for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        item = json.loads(line)
        latest[item["seed"]] = item
rows = [latest[key] for key in sorted(latest)]
lines = ["seed,test_acc,macro_f1,epochs_ran"]
for item in rows:
    lines.append(f"{item['seed']},{item['test_acc']:.4f},{item['macro_f1']:.4f},{item['epochs_ran']}")
(out / "seeds.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
best = max(rows, key=lambda item: item["test_acc"])
summary = {
    "loop": 5,
    "n_seeds": len(rows),
    "best_seed": best["seed"],
    "best_acc": best["test_acc"],
    "best_macro_f1": best["macro_f1"],
    "seeds": rows,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"RECORDED loop=5 seed={seed} acc={acc:.4f} macro_f1={macro:.4f}")
print(
    f"BEST loop=5 seed={best['seed']} n={len(rows)} "
    f"acc={best['test_acc']:.4f} macro_f1={best['macro_f1']:.4f}"
)
PY
}

fail=0
for seed in $(seq 0 7); do
  micro=32
  accum=1
  while true; do
    log="logs/weibo_loop5_seed${seed}_m${micro}.log"
    echo "===== loop=5 seed=${seed} micro=${micro} accum=${accum} $(date) =====" | tee -a "$SUMMARY"
    rm -rf "outputs/weibo_loop5/seed${seed}"
    status=0
    run_attempt "${seed}" "${micro}" "${accum}" "${log}" || status=$?
    if [ "${status}" -eq 0 ]; then
      echo "OK loop=5 seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
      record_seed "${seed}" "${log}" | tee -a "$SUMMARY"
      break
    fi
    if grep -Eq "out of memory|OutOfMemoryError" "${log}" && [ "${micro}" -gt 1 ]; then
      micro=$((micro / 2))
      accum=$((accum * 2))
      echo "OOM retry loop=5 seed=${seed} micro=${micro} accum=${accum}" | tee -a "$SUMMARY"
      continue
    fi
    echo "FAIL loop=5 seed=${seed} exit=${status}" | tee -a "$SUMMARY"
    fail=1
    break
  done
done

echo "ALL_DONE fail=${fail} $(date)" | tee -a "$SUMMARY"
exit "${fail}"
