#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_vitt
exec > >(tee -a logs/pod_entry_vitt.log) 2>&1
echo "[vitt] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[vitt] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
python train.py --dataset waterbirds --method vitbt_sdrop_gap --self_gamma --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 64 --num_workers 12 --seed 99 --amp --save_dir ./checkpoints_smoke --subset_frac 0.1 || { echo "[vitt] 스모크 실패"; terminate; exit 1; }
rm -rf ./checkpoints_smoke
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
CKPT=./checkpoints_vitt nohup bash run_vitt.sh > logs/run_vitt_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[vitt] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[vitt] 완료 $(ls checkpoints_vitt/*_history.csv 2>/dev/null | wc -l)"; terminate
