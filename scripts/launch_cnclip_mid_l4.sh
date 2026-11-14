#!/bin/bash
# Wait until the three Weibo tower jobs finish, then start loop-4 mid-training.
set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs

if [[ -z "${SWANLAB_API_KEY:-}" && -f logs/swanlab_api_key ]]; then
    SWANLAB_API_KEY=$(tr -d '[:space:]' < logs/swanlab_api_key)
    export SWANLAB_API_KEY
fi
: "${SWANLAB_API_KEY:?export SWANLAB_API_KEY first}"
# SwanLab reads SWANLAB_PROJECT as a JSON settings field. A plain name breaks login.
unset SWANLAB_PROJECT SWANLAB_EXP SWANLAB_MODE SWANLAB_WORKSPACE

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
"$PY" - << 'PY'
import os
import swanlab
swanlab.login(api_key=os.environ["SWANLAB_API_KEY"], save=False)
print("swanlab login ok", flush=True)
PY

for id in 147841 147842 147843; do
    while squeue -j "$id" -h 2>/dev/null | grep -q .; do
        echo "waiting weibo job $id $(date '+%F %T')"
        sleep 60
    done
    state=$(sacct -j "$id" -n -X -o State | awk 'NR==1 {print $1}')
    echo "weibo job $id ${state:-unknown}"
    if [[ "$state" != "COMPLETED" ]]; then
        echo "stop: job $id ended as ${state:-unknown}"
        exit 1
    fi
done

saved_key="$SWANLAB_API_KEY"
unset SWANLAB_API_KEY
jid=$(sbatch --parsable scripts/train_cnclip_mid_l4.sbatch)
export SWANLAB_API_KEY="$saved_key"
unset saved_key
echo "submitted cnclip-mid-l4 $jid"

nohup "$PY" scripts/watch_cnclip_l4.py \
    --train-job "$jid" \
    --train-log "logs/cnclip-mid-l4-${jid}.out" \
    > "logs/watch-cnclip-l4-${jid}.log" 2>&1 &
echo "watcher pid $! log logs/watch-cnclip-l4-${jid}.log"
