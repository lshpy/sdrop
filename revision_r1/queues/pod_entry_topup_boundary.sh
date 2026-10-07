#!/usr/bin/env bash
# Seed top-up, NICO++ / MetaShift / BAR / Colored-MNIST (run_topup_boundary.sh, 50 runs, ~2.5 GPU-h). Sharded by SHARD/NSHARD (env).
# Datasets are normally on the volume; each is fetched only if missing, with the original pod entry's code. If anything has to be
# downloaded, run this pod ALONE: a download that fills the volume kills every other pod's queue (2026-09-22).
set -u
MAX_HOURS="${MAX_HOURS:-4}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
cd /workspace/sdrop || exit 1; mkdir -p logs analysis data
TAG="topup_boundary_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
free=$(df -Pk /workspace | awk 'NR==2{print int($4/1048576)}'); echo "[$TAG] volume free ${free} GB"
[ "$free" -lt "$MIN_FREE_GB" ] && { echo "[$TAG] less than ${MIN_FREE_GB} GB free, abort"; exit 1; }
# NICO++ (pod_entry_screen.sh)
if [ ! -f data/nicopp/.done_nicopp ]; then
  echo "[$TAG] NICO++ download"; mkdir -p data/nicopp
  curl -L -o data/nicopp/NICOpp.zip "https://huggingface.co/datasets/2018LZY/NICOplusplus/resolve/main/NICOpp.zip" || { echo "[$TAG] NICO++ download failed"; exit 1; }
  python3 -m zipfile -e data/nicopp/NICOpp.zip data/nicopp/ && rm -f data/nicopp/NICOpp.zip && touch data/nicopp/.done_nicopp
fi
# MetaShift (pod_entry_metashift.sh)
if [ ! -d data/metashifts/MetaDatasetCatDog/train ]; then
  echo "[$TAG] MetaShift download"
  curl -L -o data/metashifts.zip "https://drive.usercontent.google.com/download?id=1WySOxBRkxAUlSokgZrC-0JaWZwcG5UMT&export=download&confirm=t" || { echo "[$TAG] MetaShift download failed"; exit 1; }
  (cd data && python3 -m zipfile -e metashifts.zip . && rm -f metashifts.zip && touch .done_metashift)
fi
# BAR (pod_entry_cmnist_bar.sh). Colored-MNIST builds on torchvision MNIST, which dataset.py downloads into data/ if absent.
[ -d data/BAR/train ] || { git clone -q --depth 1 https://github.com/alinlab/BAR.git data/BAR_repo && (cd data && (mv BAR_repo BAR 2>/dev/null || true)); }
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=30m nohup bash run_topup_boundary.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs finished on this shard"; sleep 300
done
echo "[$TAG] queue finished: $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs on this shard"
# tab:predict diagnostic ratios over the six ERM dumps (NICO++ pairs; Colored-MNIST 95% / 98%). Cheap, idempotent.
for g in nicopp_none nicopp_lion-kangaroo_grass-water_none nicopp_sheep-kangaroo_grass-water_none; do
  [ "$(ls checkpoints_screen/${g}_rate0.1_none_ep30_seed*_dump.npz 2>/dev/null | wc -l)" -eq 6 ] && \
    python analyze_selectivity.py "checkpoints_screen/${g}_rate0.1_none_ep30_seed*_dump.npz" > analysis/selectivity_${g}_topup.txt 2>&1
done
for b in 0.95 0.98; do
  [ "$(ls checkpoints_cb2/cmnist_${b}_none_rate0.1_none_ep20_seed*_dump.npz 2>/dev/null | wc -l)" -eq 6 ] && \
    python analyze_selectivity.py "checkpoints_cb2/cmnist_${b}_none_rate0.1_none_ep20_seed*_dump.npz" > analysis/selectivity_cmnist_${b}_topup.txt 2>&1
done
echo "[$TAG] done $(date '+%F %T')"
