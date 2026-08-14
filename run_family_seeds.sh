#!/usr/bin/env bash
# 가족 스윕 승자(beta=2)와 대조군(beta=0)의 시드 증량 — 짝지은 5시드 비교용.
#
#   bash run_family_seeds.sh          # 시드 1..4 x {beta0, beta2}, 8런
#
# seed 0 곡선: b0 43.23 / b1 43.39 / b2 43.77 / b4 43.39 (역U, 정점 beta=2).
# beta=2 vs beta=0 짝지은 비교가 개정 수식의 "선택 강도" 주장을 5시드로 굳힌다.
# 모든 런의 실효율은 순위 정규화 항등식으로 0.1 고정.
set -u
CKPT=./checkpoints
mkdir -p "$CKPT" logs
AMP_FLAG=""
[ "${AMP:-1}" = "1" ] && AMP_FLAG="--amp"
NW_FLAG="--num_workers ${NW:-0}"

run () {
  local id="$1"; shift
  if [ -f "$CKPT/${id}_history.csv" ]; then echo "[건너뜀] $id"; return; fi
  echo "──────────────────────────────────────────────────────────"
  echo "[실행] $id      $(date '+%F %T')"
  python train.py "$@" $AMP_FLAG $NW_FLAG 2>&1 | tee "logs/${id}.log"
  echo "[완료] $id      $(date '+%F %T')"
}

BASE="--dataset cifar100_lt --imb_ratio 100 --epochs 200 \
      --method sdrop --drop_rate 0.1 --layers L4 \
      --peakedness entropy --norm rank --mix 0.5"

for s in 1 2 3 4; do
  for b in 0 2; do
    run "cifar100_lt_sdrop_rate0.1_L4_imb100_pkentropy_nmrank_b${b}_mix0.5_seed${s}" \
        $BASE --beta $b --seed $s
  done
done

echo
echo "##################### 종료 $(date '+%F %T') #####################"
