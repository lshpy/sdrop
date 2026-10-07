#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs results checkpoints_extra
exec > >(tee -a logs/pod_entry_extra_wb.log) 2>&1
echo "[extra_wb] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[extra_wb] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
python train.py --dataset waterbirds --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 64 --num_workers 6 --seed 99 --amp --save_dir ./checkpoints_smoke_sweep --select topk || { echo "[extra_wb] 스모크 실패"; exit 1; }
rm -rf ./checkpoints_smoke_sweep
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=40m CKPT=./checkpoints_extra nohup bash run_extra_wb.sh > logs/run_extra_wb_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[extra_wb] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[extra_wb] 완료 $(ls checkpoints_extra/*_history.csv 2>/dev/null | wc -l)/12"
