#!/usr/bin/env bash
# Neurocomputing revision runs (run_neucom8.sh, 6 runs (mass-matched random control) x CIFAR-100 10%). Sharded by SHARD/NSHARD (env).
# CIFAR-100 is fetched by dataset.py (HF parquet mirror) into data/ if absent.
set -u
MAX_HOURS="${MAX_HOURS:-5}"
cd /workspace/sdrop || exit 1; mkdir -p logs data checkpoints
TAG="neucom8_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
free=$(df -Pk /workspace | awk 'NR==2{print int($4/1048576)}'); echo "[$TAG] volume free ${free} GB"
[ "$free" -lt 3 ] && { echo "[$TAG] less than 3 GB free, abort"; exit 1; }
# fetch CIFAR-100 once, before the queue, so the two shards do not race on the same files
[ -f data/cifar-100-python/meta ] || python -c "from dataset import _ensure_cifar100; _ensure_cifar100('./data')" || { echo "[$TAG] CIFAR-100 fetch failed"; exit 1; }
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=30m nohup bash run_neucom8.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs finished on this shard"; sleep 300
done
echo "[$TAG] done $(date '+%F %T'): $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs on this shard"
