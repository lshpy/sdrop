#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-8}"
cd /workspace/sdrop || exit 1; mkdir -p logs results
exec > >(tee -a logs/pod_entry_celeba.log) 2>&1
echo "[celeba] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[celeba] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
if [ ! -f data/celeba/test-00000-of-00001.parquet ]; then
  echo "[celeba] 다운로드"; python - <<'PY' || { echo "[celeba] 다운로드 실패"; terminate; exit 1; }
from huggingface_hub import snapshot_download; import shutil, pathlib
d=snapshot_download('tpremoli/CelebA-attrs', repo_type='dataset', allow_patterns='data/*', local_dir='data/_celeba_tmp')
dst=pathlib.Path('data/celeba'); dst.mkdir(exist_ok=True)
for p in pathlib.Path(d,'data').glob('*.parquet'): shutil.copy(p, dst/p.name)
print('ok', sorted(x.name for x in dst.glob('*.parquet')))
PY
fi
python train.py --dataset celeba --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 1 --lr 0.001 --batch_size 128 --num_workers 14 --seed 99 --amp --save_dir ./checkpoints_smoke --subset_frac 0.05 || { echo "[celeba] 스모크 실패"; terminate; exit 1; }
rm -rf ./checkpoints_smoke
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
nohup bash run_celeba.sh > logs/run_celeba_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[celeba] ${MAX_HOURS}h 초과"; kill "$QPID"; break; }; sleep 120; done
echo "[celeba] 완료 $(ls checkpoints/celeba_*_history.csv 2>/dev/null | wc -l)"; terminate
