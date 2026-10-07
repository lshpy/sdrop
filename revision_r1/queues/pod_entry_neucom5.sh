#!/usr/bin/env bash
# Waterbirds top-k seeds 6-13 (run_neucom5.sh, 4 ERM runs). Needs data/waterbirds on the volume.
set -u
MAX_HOURS="${MAX_HOURS:-5}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_rev
TAG="neucom5"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
[ -f data/waterbirds/test-00000-of-00001.parquet ] || { echo "[$TAG] Waterbirds missing on the volume"; ls data/waterbirds | head; exit 1; }
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=40m CKPT=./checkpoints_rev nohup bash run_neucom5.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs finished"; sleep 300
done
echo "[$TAG] done $(date '+%F %T'): $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs"
