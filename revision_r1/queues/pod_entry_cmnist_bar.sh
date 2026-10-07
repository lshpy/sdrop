#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-4}"
cd /workspace/sdrop || exit 1; mkdir -p logs results checkpoints_cb2 data
exec > >(tee -a logs/pod_entry_cmnist_bar.log) 2>&1
echo "[cmb] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[cmb] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
[ -d data/BAR/train ] || { git clone -q --depth 1 https://github.com/alinlab/BAR.git data/BAR_repo && (cd data && ls BAR_repo && (mv BAR_repo BAR 2>/dev/null || true)); }
ls data/BAR; ls data/BAR/train 2>/dev/null | head; ls data/BAR/test 2>/dev/null | head
python train.py --dataset cmnist_0.95 --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.1 --batch_size 128 --num_workers 6 --seed 99 --amp --save_dir ./checkpoints_smoke_cmb --subset_frac 0.1 --dump_feats || { echo "[cmb] 스모크 실패"; exit 1; }
python train.py --dataset bar --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 64 --num_workers 6 --seed 99 --amp --save_dir ./checkpoints_smoke_cmb --dump_feats || { echo "[cmb] BAR 스모크 실패"; exit 1; }
rm -rf ./checkpoints_smoke_cmb
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=30m CKPT=./checkpoints_cb2 nohup bash run_cmnist_bar.sh > logs/run_cmnist_bar_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[cmb] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[cmb] 완료 $(ls checkpoints_cb2/*_history.csv 2>/dev/null | wc -l)/51"
