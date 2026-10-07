#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-12}"
cd /workspace/sdrop || exit 1; mkdir -p logs results checkpoints_rev
exec > >(tee -a logs/pod_entry_rev_wb.log) 2>&1
echo "[rev_wb] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[rev_wb] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
python train.py --dataset waterbirds --method sdrop_gap --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 64 --num_workers 12 --seed 99 --amp --save_dir ./checkpoints_smoke_wb --subset_frac 0.1 --dump_feats || { echo "[rev_wb] 스모크 실패"; terminate; exit 1; }
rm -rf ./checkpoints_smoke_wb
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
CKPT=./checkpoints_rev nohup bash run_revision_wb.sh > logs/run_revision_wb_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[rev_wb] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[rev_wb] 완료 $(ls checkpoints_rev/*_history.csv 2>/dev/null | wc -l)/24"; terminate
