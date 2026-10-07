#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-5}"
cd /workspace/sdrop || exit 1; mkdir -p logs results checkpoints_screen data
exec > >(tee -a logs/pod_entry_screen2.log) 2>&1
echo "[screen2] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[screen2] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
cd data
# NICO++ is normally already extracted by pod_entry_screen.sh (marker data/nicopp/.done_nicopp); download only if missing.
if [ ! -f nicopp/.done_nicopp ]; then
  echo "[screen2] NICO++ 다운로드 $(date '+%T')"
  mkdir -p nicopp
  curl -L -o nicopp/NICOpp.zip "https://huggingface.co/datasets/2018LZY/NICOplusplus/resolve/main/NICOpp.zip" || { echo "[screen2] NICO++ 다운로드 실패"; exit 1; }
  python3 -m zipfile -e nicopp/NICOpp.zip nicopp/ && rm -f nicopp/NICOpp.zip && touch nicopp/.done_nicopp; echo "[screen2] NICO++ 압축 해제 완료 $(date '+%T')"
fi
for c in lion kangaroo sheep umbrella tent; do echo "$c grass $(ls nicopp/NICOpp/grass/$c | wc -l) water $(ls nicopp/NICOpp/water/$c | wc -l)"; done
cd /workspace/sdrop
python train.py --dataset "nicopp:lion,kangaroo:grass,water" --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 64 --num_workers 6 --seed 99 --amp --save_dir ./checkpoints_smoke_screen2 --subset_frac 0.05 --dump_feats || { echo "[screen2] 스모크 실패"; exit 1; }
rm -rf ./checkpoints_smoke_screen2
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=50m CKPT=./checkpoints_screen nohup bash run_screen2.sh > logs/run_screen2_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[screen2] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[screen2] 완료 $(ls checkpoints_screen/nicopp_*-*_grass-water_*_history.csv 2>/dev/null | wc -l)/45"
