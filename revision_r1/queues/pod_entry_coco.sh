#!/usr/bin/env bash
# COCO small-object cat/dog (prereg/PREREG_coco_smallobj_2026-09-29.md): 6 arms x 12 seeds. Shard 0 preps; others wait.
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_coco data
TAG="coco_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow pycocotools 2>&1 | tail -1
if [ "${SHARD:-0}" = "0" ]; then
  if [ ! -f data/coco_catdog/.done ]; then
    python prep_coco_smallobj.py --fetch --ann data/coco_ann && python prep_coco_smallobj.py --ann data/coco_ann --out data/coco_catdog --save-masks && touch data/coco_catdog/.done || { echo "[$TAG] prep failed"; exit 1; }
  fi
else
  for i in $(seq 1 60); do [ -f data/coco_catdog/.done ] && break; sleep 30; done
  [ -f data/coco_catdog/.done ] || { echo "[$TAG] prep not ready"; exit 1; }
fi
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
DATASETS="coco_catdog" SEEDS="0 1 2 3 4 5 6 7 8 9 10 11" CKPT=./checkpoints_coco RUN_TIMEOUT=30m nohup bash run_field.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(ls checkpoints_coco/*_history.csv 2>/dev/null | wc -l)/72"; sleep 300
done
if [ "${SHARD:-0}" = "0" ]; then
  f=checkpoints_coco/coco_catdog_none_rate0.1_none_ep30_seed0_best.pth
  for i in $(seq 1 30); do [ -f "$f" ] && break; sleep 60; done
  [ -f "$f" ] && python analyze_units.py "$f" checkpoints_coco/coco_units.csv --dataset coco_catdog || echo "[$TAG] units failed"
  [ -f "$f" ] && python analyze_bgshare.py "$f" data/coco_catdog/test-00000-of-00001.parquet data/coco_catdog/test_masks.npz checkpoints_coco/coco_bgshare.csv || echo "[$TAG] bgshare failed"
fi
echo "[$TAG] done $(ls checkpoints_coco/*_history.csv 2>/dev/null | wc -l)/72"
