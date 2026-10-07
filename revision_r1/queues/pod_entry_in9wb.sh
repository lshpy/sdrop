#!/usr/bin/env bash
# Entry for prereg/PREREG_in9wb_2026-10-01.md: build data/in9wb from the IN-9 release, then run_in9wb.sh.
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_in9wb data
exec > >(tee -a logs/pod_entry_in9wb.log) 2>&1
echo "[in9wb] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[in9wb] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
if [ ! -f data/in9wb/.done ]; then
  mkdir -p data/raw_in9s
  curl -L -m 1800 --retry 5 -o data/raw_in9s/bg.tar.gz https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz || { echo "[in9wb] download failed"; exit 1; }
  python prep_in9bg.py --tar data/raw_in9s/bg.tar.gz --out data/in9wb --classes 0,1 --bgs 2,8 --rho 0.95 && touch data/in9wb/.done || { echo "[in9wb] prep failed"; exit 1; }
  rm -rf data/raw_in9s
fi
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=30m CKPT=./checkpoints_in9wb nohup bash run_in9wb.sh > logs/run_in9wb_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[in9wb] ${MAX_HOURS}h 초과"; pkill -P "$QPID" 2>/dev/null; kill "$QPID"; break; }; sleep 120; done
echo "[in9wb] 완료 $(ls checkpoints_in9wb/*_history.csv 2>/dev/null | wc -l)/18"
