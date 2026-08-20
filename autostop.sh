#!/usr/bin/env bash
# 큐가 끝나면 결과를 묶고 Pod을 스스로 Stop시킨다 (GPU 과금 중단, /workspace 보존).
#
#   nohup bash autostop.sh > logs/autostop.log 2>&1 &
#
# MAX_HOURS 안에 안 끝나면 (행이 걸렸다고 보고) 그때도 Stop한다 — 밤새 과금 방지.
set -u
MAX_HOURS="${MAX_HOURS:-8}"
cd "$(dirname "$0")"
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))

echo "[autostop] 감시 시작. 최대 ${MAX_HOURS}시간."

while true; do
  # run_rented.sh 프로세스가 하나도 없으면 큐 종료로 판단
  if ! pgrep -f "run_rented.sh" > /dev/null 2>&1; then
    echo "[autostop] 큐 종료 감지 $(date '+%F %T')"; break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[autostop] ${MAX_HOURS}시간 초과 — 강제 종료 $(date '+%F %T')"; break
  fi
  sleep 60
done

echo "[autostop] 결과 수집 중..."
bash collect_results.sh || echo "[autostop] 수집 실패(계속 진행)"

echo "[autostop] 완료된 런:"
ls checkpoints/*_history.csv 2>/dev/null | wc -l

if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl > /dev/null 2>&1; then
  echo "[autostop] Pod $RUNPOD_POD_ID 를 Stop합니다."
  sleep 10
  runpodctl stop pod "$RUNPOD_POD_ID"
else
  echo "[autostop] ⚠ runpodctl 또는 RUNPOD_POD_ID 없음 — 수동으로 Stop하세요."
fi
