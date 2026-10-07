#!/usr/bin/env bash
# Object-size experiment (prereg/PREREG_objsize_2026-09-29.md): in9bg with the object scaled x0.5 and x0.3.
# Shard 0 preps both scales (same fg/bg pairs as in9bg); other shards wait for the .done markers.
set -u
MAX_HOURS="${MAX_HOURS:-4}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_in9scale data
TAG="in9scale_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
if [ "${SHARD:-0}" = "0" ]; then
  if [ ! -f data/in9bg_s05/.done ] || [ ! -f data/in9bg_s03/.done ]; then
    mkdir -p data/raw_in9s
    curl -L -m 1800 --retry 5 -o data/raw_in9s/bg.tar.gz https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz || { echo "[$TAG] download failed"; exit 1; }
    for sc in 05 03; do
      python prep_in9bg.py --tar data/raw_in9s/bg.tar.gz --out data/in9bg_s$sc --scale 0.${sc#0} && touch data/in9bg_s$sc/.done || { echo "[$TAG] prep failed $sc"; exit 1; }
    done
    rm -rf data/raw_in9s
  fi
else
  for i in $(seq 1 60); do [ -f data/in9bg_s05/.done ] && [ -f data/in9bg_s03/.done ] && break; sleep 30; done
  [ -f data/in9bg_s03/.done ] || { echo "[$TAG] prep not ready"; exit 1; }
fi
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
DATASETS="in9bg_s05 in9bg_s03" CKPT=./checkpoints_in9scale RUN_TIMEOUT=40m nohup bash run_field.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(ls checkpoints_in9scale/*_history.csv 2>/dev/null | wc -l)/72"; sleep 300
done
if [ "${SHARD:-0}" = "0" ]; then
  for sc in 05 03; do f=checkpoints_in9scale/in9bg_s${sc}_none_rate0.1_none_ep30_seed0_best.pth
    for i in $(seq 1 40); do [ -f "$f" ] && break; sleep 60; done
    [ -f "$f" ] && python analyze_units.py "$f" checkpoints_in9scale/in9scale_units.csv --dataset in9bg_s$sc || echo "[$TAG] units failed $sc"
  done
fi
echo "[$TAG] done $(ls checkpoints_in9scale/*_history.csv 2>/dev/null | wc -l)/72"
