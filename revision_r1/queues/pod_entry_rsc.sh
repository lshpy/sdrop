#!/usr/bin/env bash
# Entry for prereg/PREREG_in9wb_2026-10-01.md: build data/in9wb from the IN-9 release, then run_rsc.sh.
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_rsc data
exec > >(tee -a logs/pod_entry_rsc.log) 2>&1
echo "[rsc] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[rsc] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=30m CKPT=./checkpoints_rsc nohup bash run_rsc.sh > logs/run_rsc_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[rsc] ${MAX_HOURS}h 초과"; pkill -P "$QPID" 2>/dev/null; kill "$QPID"; break; }; sleep 120; done
echo "[rsc] 완료 $(ls checkpoints_rsc/*_history.csv 2>/dev/null | wc -l)/12"
