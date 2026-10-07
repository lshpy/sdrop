#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_mech checkpoints_extra
exec > >(tee -a logs/pod_entry_rest.log) 2>&1
echo "[rest] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[rest] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
( RUN_TIMEOUT=40m CKPT=./checkpoints_mech bash run_mech_wb.sh; RUN_TIMEOUT=40m CKPT=./checkpoints_extra bash run_extra_wb.sh ) > logs/run_rest_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[rest] ${MAX_HOURS}h 초과"; pkill -P "$QPID"; kill "$QPID"; break; }; sleep 120; done
echo "[rest] 완료 mech $(ls checkpoints_mech/*_history.csv 2>/dev/null | wc -l)/42 extra $(ls checkpoints_extra/*_history.csv 2>/dev/null | wc -l)/12"
