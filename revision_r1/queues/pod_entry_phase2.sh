#!/usr/bin/env bash
# 파드 C: Waterbirds → Table1 SE/CBAM 재실행 → CUB 바운딩박스 귀속 → Terminate.
set -u
MAX_HOURS="${MAX_HOURS:-30}"
cd /workspace/sdrop || exit 1
mkdir -p logs results
exec > >(tee -a logs/pod_entry_phase2.log) 2>&1
echo "[phase2] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () {
  echo "[phase2] Terminate $(date '+%F %T')"; sync
  curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo
}
pip install -q numpy pandas matplotlib scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
nvidia-smi --query-gpu=name --format=csv,noheader

# 1) Waterbirds 데이터 (없으면 HF 에서)
if [ ! -f data/waterbirds/test-00000-of-00001.parquet ]; then
  echo "[phase2] Waterbirds 다운로드"
  python - <<'PY' || { echo "[phase2] 다운로드 실패"; }
from huggingface_hub import snapshot_download
import shutil, pathlib
d = snapshot_download('grodino/waterbirds', repo_type='dataset', allow_patterns='data/*', local_dir='data/_wb_tmp')
dst = pathlib.Path('data/waterbirds'); dst.mkdir(exist_ok=True)
for p in pathlib.Path(d, 'data').glob('*.parquet'): shutil.copy(p, dst / p.name)
print('ok', sorted(x.name for x in dst.glob('*.parquet')))
PY
fi
# 2) 무효 SE/CBAM 결과 격리 (A3 버그)
mkdir -p checkpoints/invalid_A3
for f in checkpoints/cifar100_senet_rate0.0_L4_seed*_* checkpoints/cifar100_cbam_rate0.0_L4_seed*_*; do
  [ -e "$f" ] && mv "$f" checkpoints/invalid_A3/ && echo "[phase2] 격리: $(basename $f)"
done
# 3) 스모크: waterbirds 1ep
if [ -f data/waterbirds/test-00000-of-00001.parquet ]; then
  python train.py --dataset waterbirds --method sdrop_shuffle --drop_rate 0.1 --layers L3 --epochs 1 --lr 0.001 \
     --batch_size 64 --num_workers 12 --seed 99 --amp --save_dir ./checkpoints_smoke || { echo "[phase2] 스모크 실패"; terminate; exit 1; }
  rm -rf ./checkpoints_smoke
fi
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
( [ -f data/waterbirds/test-00000-of-00001.parquet ] && bash run_waterbirds.sh
  bash run_table1_fix.sh
  python analyze_cub_bbox.py --n 500
  bash run_protocol_val.sh
  python analyze_cub_bbox.py --n 500 ) > logs/phase2_queue.log 2>&1 &
QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[phase2] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }
  sleep 120
done
echo "[phase2] 완료 protocol=$(ls checkpoints/cifar100_*_val0.1_*_history.csv 2>/dev/null | wc -l) waterbirds=$(ls checkpoints/waterbirds_*_history.csv 2>/dev/null | wc -l) table1=$(ls checkpoints/cifar100_{senet,cbam}_rate0.0_L4_seed*_history.csv 2>/dev/null | wc -l)"
terminate
