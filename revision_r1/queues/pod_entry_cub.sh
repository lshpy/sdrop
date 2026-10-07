#!/usr/bin/env bash
# 파드 시작 명령(dockerStartCmd)에서 호출됨. 끝나면 스스로 Terminate.
#   - 볼륨(/workspace)에 결과가 남으므로 Terminate 가 안전 (Stop 은 디스크 과금 지속)
#   - MAX_HOURS 초과 시에도 Terminate — 밤새 과금 방지
set -u
MAX_HOURS="${MAX_HOURS:-16}"
QUEUE="${QUEUE:-run_cub.sh}"          # run_cub.sh | run_cub_guided.sh
SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SMOKE_METHOD="${SMOKE_METHOD:-sdrop_shuffle}"
cd /workspace/sdrop || { echo "볼륨 없음"; exit 1; }
mkdir -p logs
exec > >(tee -a logs/pod_entry_cub.log) 2>&1
echo "[entry] 시작 $(date '+%F %T')  pod=${RUNPOD_POD_ID:-?}  queue=$QUEUE shard=$SHARD/$NSHARD"

terminate () {
  echo "[entry] Terminate 요청 $(date '+%F %T')"
  python summarize.py >/dev/null 2>&1 || true
  sync
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"
    echo
  else
    echo "[entry] ⚠ API 키/POD_ID 없음 — 수동 종료 필요"
  fi
}

pip install -q numpy pandas matplotlib scikit-learn tqdm pyarrow pillow 2>&1 | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.cuda.is_available())"
[ -d data/CUB_200_2011/images ] || { echo "[entry] CUB 데이터 없음"; terminate; exit 1; }

# 스모크: 새 팔(shuffle) 1에폭. 여기서 죽으면 큐 돌리지 않는다.
if [ "$SKIP_SMOKE" != "1" ]; then
  echo "[entry] 스모크 테스트 ($SMOKE_METHOD)"
  if ! python train.py --dataset cub200 --method "$SMOKE_METHOD" --drop_rate 0.1 --layers L3 \
        --epochs 1 --lr 0.01 --batch_size 64 --num_workers 12 --seed 99 --amp --save_dir ./checkpoints_smoke; then
    echo "[entry] 스모크 실패 — 종료"; terminate; exit 1
  fi
  rm -rf ./checkpoints_smoke
fi

deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
SHARD=$SHARD NSHARD=$NSHARD nohup bash "$QUEUE" > "logs/${QUEUE%.sh}_queue.log" 2>&1 &
QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[entry] ${MAX_HOURS}h 초과 — 강제 종료"; kill "$QPID"; break; }
  sleep 120
done
echo "[entry] 완료된 CUB 런: $(ls checkpoints/cub200_*_history.csv 2>/dev/null | wc -l)"
terminate
