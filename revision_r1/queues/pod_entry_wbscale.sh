#!/usr/bin/env bash
# Waterbirds rescaled (prereg/PREREG_wbscale_2026-09-29.md). Shard 0 fetches Places + segmentations and builds 4 scales.
set -u
MAX_HOURS="${MAX_HOURS:-4}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_wbscale data
TAG="wbscale_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
ALL="data/wbscale_s05/.done data/wbscale_s10/.done data/wbscale_s15/.done data/wbscale_s20/.done"
if [ "${SHARD:-0}" = "0" ]; then
  [ -f data/places_wb/.done ] || { python prep_waterbirds_scaled.py fetch-places --out data/places_wb && touch data/places_wb/.done; } || { echo "[$TAG] places failed"; exit 1; }
  [ -s data/segmentations.tgz ] || curl -L -m 1800 --retry 5 -o data/segmentations.tgz https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz || { echo "[$TAG] seg failed"; exit 1; }
  for sc in 0.5 1.0 1.5 2.0; do d=data/wbscale_s$(echo $sc | tr -d .); [ "$sc" = "0.5" ] && d=data/wbscale_s05
    [ -f $d/.done ] || { python prep_waterbirds_scaled.py build --scale $sc --out $d && touch $d/.done; } || { echo "[$TAG] build failed $sc"; exit 1; }
  done
else
  for i in $(seq 1 120); do ok=1; for f in $ALL; do [ -f $f ] || ok=0; done; [ $ok = 1 ] && break; sleep 30; done
  for f in $ALL; do [ -f $f ] || { echo "[$TAG] prep not ready ($f)"; exit 1; }; done
fi
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
CKPT=./checkpoints_wbscale RUN_TIMEOUT=30m nohup bash run_wbscale.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(ls checkpoints_wbscale/*_history.csv 2>/dev/null | wc -l)/144"; sleep 300
done
if [ "${SHARD:-0}" = "0" ]; then
  for sc in 05 10 15 20; do f=checkpoints_wbscale/wbscale_s${sc}_none_rate0.1_none_ep30_seed20_best.pth
    [ -f "$f" ] && python analyze_bgshare.py "$f" data/wbscale_s${sc}/test-00000-of-00001.parquet data/wbscale_s${sc}/test_masks.npz checkpoints_wbscale/wbscale_bgshare.csv || echo "[$TAG] bgshare failed $sc"
  done
fi
echo "[$TAG] done $(ls checkpoints_wbscale/*_history.csv 2>/dev/null | wc -l)/144"
