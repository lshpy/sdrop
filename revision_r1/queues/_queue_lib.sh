#!/usr/bin/env bash
# 큐 스크립트 공용 헬퍼. 각 run_*.sh 맨 위에서:  source "$(dirname "$0")/_queue_lib.sh"
#   run <run_id> <train.py 인자...>   — history.csv 가 있으면 건너뜀, SHARD/NSHARD 로 분할, 로그는 logs/<id>.log
set -u
SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"
CKPT="${CKPT:-./checkpoints}"; mkdir -p "$CKPT" logs
_QIDX=0
run () {
  local id="$1"; shift
  local mine=$(( _QIDX % NSHARD )); _QIDX=$(( _QIDX + 1 ))
  [ "$mine" != "$SHARD" ] && return
  [ -f "$CKPT/${id}_history.csv" ] && { echo "[건너뜀] $id"; return; }
  echo "[실행] $id  $(date '+%F %T')"
  timeout "${RUN_TIMEOUT:-0}" python train.py "$@" --amp 2>&1 | tee "logs/${id}.log"   # RUN_TIMEOUT=40m 등으로 행 방지 (0 = 무제한)
  echo "[완료] $id  $(date '+%F %T')"
}
