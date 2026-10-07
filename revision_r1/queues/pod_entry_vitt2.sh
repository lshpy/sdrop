#!/usr/bin/env bash
# ViT-B token/MLP SDrop queue, resume-safe (run_vitt.sh skips finished run ids). No smoke test: the code path is already validated.
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_vitt
exec > >(tee -a logs/pod_entry_vitt2.log) 2>&1
terminate () { echo "[vitt2] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[vitt2] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} done=$(ls checkpoints_vitt/*_history.csv 2>/dev/null | wc -l)/24"
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
CKPT=./checkpoints_vitt RUN_TIMEOUT=25m nohup bash run_vitt.sh > logs/run_vitt_queue2.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[vitt2] ${MAX_HOURS}h 초과"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[vitt2] 진행 $(date '+%T') $(ls checkpoints_vitt/*_history.csv 2>/dev/null | wc -l)/24"; sleep 300
done
echo "[vitt2] 완료 $(ls checkpoints_vitt/*_history.csv 2>/dev/null | wc -l)/24"
