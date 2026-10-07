#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-4}"
cd /workspace/sdrop || exit 1; mkdir -p logs results checkpoints_spw data/spawrious
exec > >(tee -a logs/pod_entry_spawrious.log) 2>&1
echo "[spw] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[spw] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
cd data/spawrious
for ch in o2o_easy o2o_hard; do
  url=$( [ $ch = o2o_easy ] && echo "https://www.dropbox.com/s/kwhiv60ihxe3owy/spawrious224__o2o_easy.tar.gz?dl=1" || echo "https://www.dropbox.com/s/p1ry121m2gjj158/spawrious224__o2o_hard.tar.gz?dl=1" )
  if [ ! -f ".done_$ch" ]; then
    echo "[spw] 다운로드 $ch $(date '+%T')"; curl -L -o "$ch.tar.gz" "$url" || { echo "[spw] 다운로드 실패 $ch"; exit 1; }
    tar -xzf "$ch.tar.gz" && rm -f "$ch.tar.gz" && touch ".done_$ch"; echo "[spw] 압축 해제 완료 $ch $(date '+%T')"
  fi
done
ls spawrious224 2>/dev/null | head; find spawrious224 -maxdepth 2 | head -20
cd /workspace/sdrop
python train.py --dataset spawrious_o2o_easy --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 64 --num_workers 8 --seed 99 --amp --save_dir ./checkpoints_smoke_spw --subset_frac 0.05 --dump_feats || { echo "[spw] 스모크 실패"; exit 1; }
rm -rf ./checkpoints_smoke_spw
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=50m CKPT=./checkpoints_spw nohup bash run_spawrious.sh > logs/run_spawrious_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[spw] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[spw] 완료 $(ls checkpoints_spw/*_history.csv 2>/dev/null | wc -l)/18"
