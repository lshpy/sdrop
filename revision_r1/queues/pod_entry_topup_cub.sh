#!/usr/bin/env bash
# Seed top-up, CUB-200-2011 every arm 3 -> 6 seeds (run_topup_cub.sh, 64 runs x ~16-18 min = ~18 GPU-h; CORE_ONLY=1: 27 runs, ~7.3 GPU-h).
# Sharded by SHARD/NSHARD (env), e.g. NSHARD=4 -> ~4.5 h per pod. CUB was uploaded to the volume by hand; there is no download path.
set -u
MAX_HOURS="${MAX_HOURS:-6}"
cd /workspace/sdrop || exit 1; mkdir -p logs
TAG="topup_cub_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1} core_only=${CORE_ONLY:-0}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
[ -d data/CUB_200_2011/images ] || { echo "[$TAG] data/CUB_200_2011 missing on the volume, abort"; exit 1; }
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=40m nohup bash run_topup_cub.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs finished on this shard"; sleep 300
done
echo "[$TAG] done $(date '+%F %T'): $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs on this shard"
# Post-hoc (local): CUB uses the final-epoch 'acc' column (bh_fdr.py metric 'final'); rerun scripts/bh_fdr.py after syncing.
