#!/usr/bin/env bash
# Screening-set backbone diagnostic (2026-09-29, PREREG_SCREEN_DATASETS.md): prep in9bg (280 MB), counteranimal
# (1.8 GB), fairface_age (0.56 GB) and, if the volume lacks it, CelebA (1.4 GB); then run_screen_diag.sh:
# one ERM seed each + analyze_units.py -> checkpoints_screen/screen_units.csv.
# Several pods/shards may start at once (SHARD/NSHARD): only SHARD 0 downloads and preps; the others wait for
# the .done markers (PREP_WAIT_MIN, default 90 min) and exit if they never appear.
set -u
MAX_HOURS="${MAX_HOURS:-4}"; SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"; PREP_WAIT_MIN="${PREP_WAIT_MIN:-90}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_screen data
exec > >(tee -a logs/pod_entry_screen_diag.shard${SHARD}.log) 2>&1
echo "[screen] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard $SHARD/$NSHARD"
terminate () { echo "[screen] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
HF=https://huggingface.co/datasets
get () { curl -fL -C - --retry 5 --connect-timeout 30 --speed-limit 100000 --speed-time 120 -o "$1" "$2" || { echo "[screen] 다운로드 실패 $2"; exit 1; }; }
DONE="data/in9bg/.done data/counteranimal/.done data/fairface_age/.done data/celeba/.done"

if [ "$SHARD" = 0 ]; then
  if [ ! -f data/in9bg/.done ]; then
    echo "[screen] IN-9 $(date '+%T')"; mkdir -p data/raw_in9
    get data/raw_in9/bg.tar.gz https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz
    python prep_in9bg.py --tar data/raw_in9/bg.tar.gz --out data/in9bg && touch data/in9bg/.done && rm -rf data/raw_in9 || { echo "[screen] in9bg prep 실패"; exit 1; }
  fi
  if [ ! -f data/counteranimal/.done ]; then
    echo "[screen] CounterAnimal $(date '+%T')"; mkdir -p data/raw_ca
    for i in 0 1 2 3; do get data/raw_ca/test-0000$i-of-00004.parquet "$HF/TMLR-Group-HF/counteranimal/resolve/e7811153bef8cf5aa274e22b9db4bf002156e376/data/test-0000$i-of-00004.parquet"; done
    python prep_counteranimal.py --src data/raw_ca --out data/counteranimal && touch data/counteranimal/.done && rm -rf data/raw_ca || { echo "[screen] counteranimal prep 실패"; exit 1; }
  fi
  if [ ! -f data/fairface_age/.done ]; then
    echo "[screen] FairFace $(date '+%T')"; mkdir -p data/raw_ff
    for f in train-00000-of-00002-d405faba4f4b9b85 train-00001-of-00002-dd3cb68164727418 validation-00000-of-00001-951dbd63c8724ee1; do
      get data/raw_ff/$f.parquet "$HF/HuggingFaceM4/FairFace/resolve/54d573cdb8b5af490ba8da9da2799628f6e5c496/0.25/$f.parquet"; done
    python prep_fairface_age.py --src data/raw_ff --out data/fairface_age && touch data/fairface_age/.done && rm -rf data/raw_ff || { echo "[screen] fairface prep 실패"; exit 1; }
  fi
  if [ ! -f data/celeba/.done ]; then                      # the volume normally has it already (Blond_Hair runs)
    mkdir -p data/celeba
    for f in train-00000-of-00003 train-00001-of-00003 train-00002-of-00003 validation-00000-of-00001 test-00000-of-00001; do
      [ -f data/celeba/$f.parquet ] || get data/celeba/$f.parquet "$HF/tpremoli/CelebA-attrs/resolve/ed9021d2871ceddbd3cf0fb642544bd7c60c5152/data/$f.parquet"; done
    touch data/celeba/.done
  fi
else
  echo "[screen] shard $SHARD: SHARD 0 의 prep 대기 (최대 ${PREP_WAIT_MIN}분)"
  t_end=$(( $(date +%s) + PREP_WAIT_MIN*60 ))
  until ls $DONE >/dev/null 2>&1; do
    [ "$(date +%s)" -ge "$t_end" ] && { echo "[screen] prep 대기 시간 초과"; exit 1; }; sleep 60
  done
fi
echo "[screen] 데이터 준비 완료 $(date '+%T')"

deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
SHARD=$SHARD NSHARD=$NSHARD RUN_TIMEOUT=100m CKPT=./checkpoints_screen nohup bash run_screen_diag.sh > logs/run_screen_diag_queue.shard${SHARD}.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[screen] ${MAX_HOURS}h 초과"; pkill -P "$QPID" 2>/dev/null; kill "$QPID"; break; }; sleep 120; done
echo "[screen] 완료 $(ls checkpoints_screen/*_history.csv 2>/dev/null | wc -l)/5"; cat checkpoints_screen/screen_units.csv 2>/dev/null
