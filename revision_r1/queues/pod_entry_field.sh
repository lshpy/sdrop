#!/usr/bin/env bash
# Pre-registered field datasets (prereg/PREREG_field_datasets_2026-09-29.md): 72 runs, sharded by SHARD/NSHARD (env).
set -u
MAX_HOURS="${MAX_HOURS:-5}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_field
TAG="field_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
CKPT=./checkpoints_field RUN_TIMEOUT=70m nohup bash run_field.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(ls checkpoints_field/*_history.csv 2>/dev/null | wc -l)/72"; sleep 300
done
echo "[$TAG] done $(ls checkpoints_field/*_history.csv 2>/dev/null | wc -l)/72"
