#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-5}"
cd /workspace/sdrop || exit 1; mkdir -p logs results checkpoints_screen data
exec > >(tee -a logs/pod_entry_screen.log) 2>&1
echo "[screen] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[screen] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
cd data
# NICO++ DG subset (HF 2018LZY/NICOplusplus, NICOpp.zip 1.37GB): NICOpp/<domain>/<category>/*.jpg. Ranged GET verified 2026-09-23.
if [ ! -f nicopp/.done_nicopp ]; then
  echo "[screen] NICO++ 다운로드 $(date '+%T')"
  mkdir -p nicopp
  curl -L -o nicopp/NICOpp.zip "https://huggingface.co/datasets/2018LZY/NICOplusplus/resolve/main/NICOpp.zip" || { echo "[screen] NICO++ 다운로드 실패"; exit 1; }
  python3 -m zipfile -e nicopp/NICOpp.zip nicopp/ && rm -f nicopp/NICOpp.zip && touch nicopp/.done_nicopp; echo "[screen] NICO++ 압축 해제 완료 $(date '+%T')"
fi
ls nicopp/NICOpp/grass nicopp/NICOpp/water | head; ls nicopp/NICOpp/grass/elephant | wc -l
cd /workspace/sdrop
for ds in nicopp; do
  python train.py --dataset $ds --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 64 --num_workers 6 --seed 99 --amp --save_dir ./checkpoints_smoke_screen --subset_frac 0.05 --dump_feats || { echo "[screen] $ds 스모크 실패"; exit 1; }
done
rm -rf ./checkpoints_smoke_screen
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=50m CKPT=./checkpoints_screen nohup bash run_screen.sh > logs/run_screen_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[screen] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[screen] 완료 $(ls checkpoints_screen/*_history.csv 2>/dev/null | wc -l)/21"
