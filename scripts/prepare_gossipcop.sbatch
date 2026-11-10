#!/bin/bash
#SBATCH --job-name=gossipcop
#SBATCH --account=lumia
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs

export http_proxy=http://192.168.102.101:7890
export https_proxy=http://192.168.102.101:7890
export no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/yxfeng/.cache/huggingface

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
DEPS=/home/yxfeng/cscwd27_loopfuse/.deps
mkdir -p "$DEPS"
if ! PYTHONPATH="$DEPS" "$PY" -c "import datasets, pyarrow" >/dev/null 2>&1; then
  "$PY" -m pip install --target "$DEPS" "datasets==2.21.0" "pyarrow==17.0.0"
fi
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/prepare_gossipcop.py
