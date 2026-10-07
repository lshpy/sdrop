#!/usr/bin/env bash
# Pre-registered backbone diagnostic on two field datasets (2026-09-29, PREREG_DATASETS.md):
# prep CCT20 (LILA, 6.5 GB tar streamed into parquet) and ISIC-artifact (Bissoto trap split 1, 1.6 GB of
# ISIC archive JPEGs), then run_prereg_diag.sh: one ERM seed each + analyze_units.py -> checkpoints_prereg/prereg_units.csv
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_prereg data
exec > >(tee -a logs/pod_entry_prereg_diag.log) 2>&1
echo "[prereg] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[prereg] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
LILA=https://storage.googleapis.com/public-datasets-lila/caltechcameratraps     # Azure mirror: https://lilawildlife.blob.core.windows.net/lila-wildlife/caltechcameratraps
if [ ! -f data/cct20/.done ]; then
  echo "[prereg] CCT20 다운로드 $(date '+%T')"; mkdir -p data/cct20_raw
  curl -L -C - --retry 5 -o data/cct20_raw/eccv_18_annotations.tar.gz "$LILA/eccv_18_annotations.tar.gz" || { echo "[prereg] CCT20 주석 실패"; exit 1; }
  curl -L -C - --retry 5 -o data/cct20_raw/eccv_18_all_images_sm.tar.gz "$LILA/eccv_18_all_images_sm.tar.gz" || { echo "[prereg] CCT20 이미지 실패"; exit 1; }
  python prep_cct20.py --tar data/cct20_raw/eccv_18_all_images_sm.tar.gz --ann data/cct20_raw/eccv_18_annotations.tar.gz --out data/cct20 \
    && touch data/cct20/.done && rm -f data/cct20_raw/eccv_18_all_images_sm.tar.gz || { echo "[prereg] CCT20 prep 실패"; exit 1; }
  echo "[prereg] CCT20 완료 $(date '+%T')"
fi
if [ ! -f data/isic_artifact/.done ]; then
  echo "[prereg] ISIC 다운로드 $(date '+%T')"
  python prep_isic_artifact.py --out data/isic_artifact && touch data/isic_artifact/.done || { echo "[prereg] ISIC prep 실패"; exit 1; }
  echo "[prereg] ISIC 완료 $(date '+%T')"
fi
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=60m CKPT=./checkpoints_prereg nohup bash run_prereg_diag.sh > logs/run_prereg_diag_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[prereg] ${MAX_HOURS}h 초과"; pkill -P "$QPID" 2>/dev/null; kill "$QPID"; break; }; sleep 120; done
echo "[prereg] 완료 $(ls checkpoints_prereg/*_history.csv 2>/dev/null | wc -l)/2"; cat checkpoints_prereg/prereg_units.csv 2>/dev/null
