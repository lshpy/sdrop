#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-4}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_mech2
exec > >(tee -a logs/pod_entry_mech2.log) 2>&1
echo "[mech2] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[mech2] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
# MetaShift / Spawrious are already on the volume (no download here).
for d in data/metashifts/MetaDatasetCatDog data/spawrious/spawrious224; do [ -d "$d" ] || echo "[mech2] 경고: $d 없음 — run_spatial 의 해당 런은 실패한다"; done
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
( RUN_TIMEOUT=50m CKPT=./checkpoints_mech2 bash run_mech2_wb.sh; RUN_TIMEOUT=50m CKPT=./checkpoints_mech2 bash run_spatial.sh ) > logs/run_mech2_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[mech2] ${MAX_HOURS}h 초과"; pkill -P "$QPID"; kill "$QPID"; break; }; sleep 120; done
echo "[mech2] 완료 $(ls checkpoints_mech2/*_history.csv 2>/dev/null | wc -l)/19"
