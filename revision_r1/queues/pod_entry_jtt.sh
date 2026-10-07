#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-4}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_jtt
exec > >(tee -a logs/pod_entry_jtt.log) 2>&1
echo "[jtt] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[jtt] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=50m CKPT=./checkpoints_jtt bash run_jtt_wb.sh > logs/run_jtt_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[jtt] ${MAX_HOURS}h 초과"; pkill -P "$QPID"; kill "$QPID"; break; }; sleep 120; done
echo "[jtt] 완료 $(ls checkpoints_jtt/*_history.csv 2>/dev/null | wc -l)/12"
