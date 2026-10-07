#!/usr/bin/env bash
# Seed top-up, Spawrious O2O-Easy / O2O-Hard seeds 3-5 (run_topup_spawrious.sh, 18 runs x ~11 min = ~3.3 GPU-h). SHARD/NSHARD (env).
# The extracted images normally sit on the volume (data/spawrious/spawrious224, markers .done_o2o_easy/.done_o2o_hard). A missing
# archive is re-downloaded (7+ GB each; run this pod ALONE then). Every PNG is verified first: a corrupt file crashed all runs on 9/22.
set -u
MAX_HOURS="${MAX_HOURS:-5}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
cd /workspace/sdrop || exit 1; mkdir -p logs analysis data/spawrious
TAG="topup_spw2_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
cd data/spawrious
for ch in o2o_easy o2o_hard; do
  [ -f ".done_$ch" ] && continue
  free=$(df -Pk . | awk 'NR==2{print int($4/1048576)}'); [ "$free" -lt "$MIN_FREE_GB" ] && { echo "[$TAG] $ch missing and only ${free} GB free, abort"; exit 1; }
  url=$( [ $ch = o2o_easy ] && echo "https://www.dropbox.com/s/kwhiv60ihxe3owy/spawrious224__o2o_easy.tar.gz?dl=1" || echo "https://www.dropbox.com/s/p1ry121m2gjj158/spawrious224__o2o_hard.tar.gz?dl=1" )
  echo "[$TAG] download $ch $(date '+%T')"; curl -L -o "$ch.tar.gz" "$url" || { echo "[$TAG] download failed $ch"; exit 1; }
  tar --no-same-owner -xzf "$ch.tar.gz" && rm -f "$ch.tar.gz" && touch ".done_$ch"
done
python3 - <<'PY' || { echo "[$TAG] corrupt Spawrious images, abort (re-extract the archive before spending GPU time)"; exit 1; }
import glob, sys; from PIL import Image
bad = []
for f in glob.glob('spawrious224/**/*.png', recursive=True):
    try: Image.open(f).verify()
    except Exception: bad.append(f)
print('[spw] corrupt files:', len(bad), bad[:5]); sys.exit(1 if bad else 0)
PY
cd /workspace/sdrop
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=50m nohup bash run_topup_spawrious.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs finished on this shard"; sleep 300
done
echo "[$TAG] done $(date '+%F %T'): $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs on this shard"
# tab:predict Spawrious ratios were computed on val+test pooled (test alone has one background per class), which
# analyze_selectivity.py does not do; recompute locally from the six ERM dumps (see TOPUP_PLAN.md).
