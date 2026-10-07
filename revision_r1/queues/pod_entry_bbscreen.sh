#!/usr/bin/env bash
# Entry for the pre-registered ResNet-101 Waterbirds robustness queue (run_bbscreen.sh). Waterbirds is already on the volume.
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_bbscreen
exec > >(tee -a logs/pod_entry_bbscreen_s${SHARD:-0}.log) 2>&1
echo "[bbscreen] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[bbscreen] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow scipy pycocotools 2>&1 | tail -1
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=40m CKPT=./checkpoints_bbscreen nohup bash run_bbscreen.sh > logs/run_bbscreen_queue_s${SHARD:-0}.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[bbscreen] ${MAX_HOURS}h 초과"; pkill -P "$QPID" 2>/dev/null; kill "$QPID"; break; }; sleep 120; done
echo "[bbscreen] 완료 $(ls checkpoints_bbscreen/*_history.csv 2>/dev/null | wc -l)/45"
