#!/usr/bin/env bash
# Seed top-up, ViT-B/16 rows of tab:vitb at 3 seeds (run_topup_vit.sh, 27 runs, ~2.5 GPU-h on a 5090). Sharded by SHARD/NSHARD (env).
# Needs only Waterbirds (on the volume) and the torchvision ViT-B/16 weights (downloaded by train.py as before).
set -u
MAX_HOURS="${MAX_HOURS:-4}"
cd /workspace/sdrop || exit 1; mkdir -p logs
TAG="topup_vit_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
[ -f data/waterbirds/test-00000-of-00001.parquet ] || { echo "[$TAG] Waterbirds missing on the volume (run pod_entry_topup_wb.sh first, it downloads it)"; exit 1; }
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=25m nohup bash run_topup_vit.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs finished on this shard"; sleep 300
done
echo "[$TAG] done $(date '+%F %T'): $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs on this shard"
# Post-hoc (local, CPU): pull histories/dumps, then scripts/bh_fdr.py (VIT-* rows) and, for the dose rows vs ERM+LA,
# scripts/analyze_vit_frontier.py against checkpoints_vitt ERM dumps. Nothing to run on the pod.
