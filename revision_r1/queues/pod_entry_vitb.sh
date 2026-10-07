#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-8}"
cd /workspace/sdrop || exit 1; mkdir -p logs results
exec > >(tee -a logs/pod_entry_vitb.log) 2>&1
echo "[vitb] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[vitb] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
python train.py --dataset waterbirds --method vitb_sdrop_energy --self_gamma --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 64 --num_workers 12 --seed 99 --amp --save_dir ./checkpoints_smoke --subset_frac 0.1 || { echo "[vitb] 스모크 실패"; terminate; exit 1; }
rm -rf ./checkpoints_smoke
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
nohup bash run_vitb.sh > logs/run_vitb_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[vitb] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[vitb] 완료 $(ls checkpoints/waterbirds_vitb_*_history.csv 2>/dev/null | wc -l)"; terminate
