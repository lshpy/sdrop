#!/usr/bin/env bash
# 무인 연쇄 실행:  현재 큐 → 코드 갱신 → 희소성 격자 → 결과 수집 → Pod Stop
#
#   curl -sO https://raw.githubusercontent.com/lshpy/sdrop/main/run_all.sh
#   nohup bash run_all.sh > logs/chain.log 2>&1 &
#
# git pull 을 지금 하지 않는 이유: 돌고 있는 큐가 train.py 를 매 런마다 새로
# 읽으므로, 도중에 기본값(--norm)이 바뀌면 한 표 안에서 설정이 섞인다.
set -u
cd "$(dirname "$0")"
mkdir -p logs
LOG(){ echo "[chain $(date '+%m-%d %H:%M')] $*"; }
GPUS="${GPUS:-2}"; PROCS="${PROCS:-8}"; MAXH="${MAXH:-30}"
deadline=$(( $(date +%s) + MAXH*3600 ))

wait_drain () {  # wait_drain <패턴> <라벨>
  while pgrep -f "$1" > /dev/null 2>&1; do
    [ "$(date +%s)" -ge "$deadline" ] && { LOG "시간 초과 — $2 대기 중단"; return 1; }
    sleep 120
  done
  LOG "$2 종료"
}

LOG "1단계 — 현재 큐(run_rented.sh) 종료 대기"
wait_drain "run_rented.sh" "run_rented"
LOG "  완료된 history: $(ls checkpoints/*_history.csv 2>/dev/null | wc -l)"

LOG "2단계 — 코드 갱신 (norm=mean 기본값, width_mult, sdrop_class, run_grid.sh)"
git pull --ff-only origin main 2>&1 | tail -3

LOG "3단계 — 희소성 격자 84런 투입 (${GPUS}GPU × ${PROCS}프로세스)"
for i in $(seq 0 $((PROCS-1))); do
  g=$(( i % GPUS ))
  SHARD=$i NSHARD=$PROCS CUDA_VISIBLE_DEVICES=$g \
    nohup bash run_grid.sh all > "logs/grid$i.log" 2>&1 &
done
sleep 60
LOG "  기동 확인: run_grid 프로세스 $(pgrep -cf run_grid.sh)개"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  /'

LOG "4단계 — 격자 종료 대기"
wait_drain "run_grid.sh" "run_grid"

LOG "5단계 — 결과 수집"
bash collect_results.sh 2>&1 | tail -3
LOG "  최종 history: $(ls checkpoints/*_history.csv 2>/dev/null | wc -l)"

LOG "6단계 — Pod Stop (GPU 과금 종료, /workspace 는 보존)"
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl > /dev/null 2>&1; then
  sleep 10
  runpodctl stop pod "$RUNPOD_POD_ID"
else
  LOG "⚠ runpodctl 없음 — 수동으로 Stop 필요"
fi
