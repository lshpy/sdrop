#!/usr/bin/env bash
# aerialwb (prereg/PREREG_aerialwb_2026-09-29.md): shard 0 downloads raw data to the CONTAINER disk (not the volume) and
# writes only the parquet dataset to the volume; other shards wait. 6 arms x 12 seeds.
set -u
MAX_HOURS="${MAX_HOURS:-4}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_aerialwb data
TAG="aerialwb_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow huggingface_hub shapely opencv-python-headless 2>&1 | tail -1
if [ "${SHARD:-0}" = "0" ]; then
  if [ ! -f data/aerialwb/.done ]; then
    RAW=/root/raw; mkdir -p $RAW && cd $RAW && python - <<'PY' || { echo "[aerialwb] download failed"; exit 1; }
from huggingface_hub import hf_hub_download as d, snapshot_download
for f in ['isaid_annotations_train.tar.gz', 'isaid_annotations_val.tar.gz']:
    d('isaaccorley/isaid', f, repo_type='dataset', revision='f5143c4756fb043186b3cc29514913d9026b0149', local_dir='isaid')
for f in ['dotav1.0_images_train.tar.gz', 'dotav1.0_images_val.tar.gz']:
    d('torchgeo/dota', f, repo_type='dataset', revision='672e63236622f7da6ee37fca44c50ac368b77cab', local_dir='dota')
snapshot_download('jonathan-roberts1/MLRSNet', repo_type='dataset', revision='e6010aa8b32734e2f2dbc11ffdeebb01f2a6858d', allow_patterns='data/*', local_dir='mlrsnet')
PY
    cd /workspace/sdrop
    python prep_aerialwb.py --isaid-train $RAW/isaid/isaid_annotations_train.tar.gz --isaid-val $RAW/isaid/isaid_annotations_val.tar.gz \
      --dota $RAW/dota/dotav1.0_images_train.tar.gz $RAW/dota/dotav1.0_images_val.tar.gz --mlrsnet "$RAW/mlrsnet/data/*" \
      --out data/aerialwb --save-masks && touch data/aerialwb/.done || { echo "[$TAG] prep failed"; exit 1; }
  fi
else
  for i in $(seq 1 120); do [ -f data/aerialwb/.done ] && break; sleep 30; done
  [ -f data/aerialwb/.done ] || { echo "[$TAG] prep not ready"; exit 1; }
fi
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
DATASETS="aerialwb" SEEDS="0 1 2 3 4 5 6 7 8 9 10 11" CKPT=./checkpoints_aerialwb RUN_TIMEOUT=30m nohup bash run_field.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(ls checkpoints_aerialwb/*_history.csv 2>/dev/null | wc -l)/72"; sleep 300
done
if [ "${SHARD:-0}" = "0" ]; then
  f=checkpoints_aerialwb/aerialwb_none_rate0.1_none_ep30_seed0_best.pth
  for i in $(seq 1 30); do [ -f "$f" ] && break; sleep 60; done
  [ -f "$f" ] && python analyze_units.py "$f" checkpoints_aerialwb/aerialwb_units.csv --dataset aerialwb || echo "[$TAG] units failed"
  [ -f "$f" ] && python analyze_bgshare.py "$f" data/aerialwb/test-00000-of-00001.parquet data/aerialwb/test_masks.npz checkpoints_aerialwb/aerialwb_bgshare.csv || echo "[$TAG] bgshare failed"
fi
echo "[$TAG] done $(ls checkpoints_aerialwb/*_history.csv 2>/dev/null | wc -l)/72"
