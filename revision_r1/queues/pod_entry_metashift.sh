#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-5}"
cd /workspace/sdrop || exit 1; mkdir -p logs results checkpoints_metashift data
exec > >(tee -a logs/pod_entry_metashift.log) 2>&1
echo "[metashift] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[metashift] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
cd data
# DISC (Wu et al. 2023) MetaShift cat/dog zip, 226MB. drive.usercontent + confirm=t skips the virus-scan page.
if [ ! -d metashifts/MetaDatasetCatDog/train ]; then
  echo "[metashift] 다운로드 $(date '+%T')"
  curl -L -o metashifts.zip "https://drive.usercontent.google.com/download?id=1WySOxBRkxAUlSokgZrC-0JaWZwcG5UMT&export=download&confirm=t" || { echo "[metashift] 다운로드 실패"; exit 1; }
  python3 -m zipfile -e metashifts.zip . && rm -f metashifts.zip && touch .done_metashift; echo "[metashift] 압축 해제 완료 $(date '+%T')"
fi
find metashifts -maxdepth 4 -type d | head -20
cd /workspace/sdrop
python train.py --dataset metashift --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 64 --num_workers 6 --seed 99 --amp --save_dir ./checkpoints_smoke_metashift --subset_frac 0.05 --dump_feats || { echo "[metashift] 스모크 실패"; exit 1; }
rm -rf ./checkpoints_smoke_metashift
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=50m CKPT=./checkpoints_metashift nohup bash run_metashift.sh > logs/run_metashift_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[metashift] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[metashift] 완료 $(ls checkpoints_metashift/*_history.csv 2>/dev/null | wc -l)/24"
