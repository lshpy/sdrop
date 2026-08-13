#!/usr/bin/env bash
# 개정 수식의 (w, beta) 2차원 가족 스윕 — CIFAR-100-LT, 시드 0
#
#   bash run_lt_family.sh
#
# 개정 수식:
#     s_c = (1-w)*rank(E_c) + w*rank(1-P~_c)        w : 무엇으로 고를지
#     p_c = p_base*(beta+1)*rank(s_c)^beta          beta : 얼마나 편파적으로
#
# 이 가족의 모서리가 기존 방법들이다:
#     beta=0        -> 모든 채널 동일 확률 = SpatialDropout (무작위 대조군)
#     w=0           -> SDropEnergy
#     w=1           -> SDropPeak
#     w=0.5,beta=1  -> SDrop 에 해당
#
# 모든 조건에서 실효 드롭률이 정확히 p_base 로 고정되므로(순위 정규화의 항등식),
# 강도 차이 없이 "선택 기준"만 비교된다. 기존 max 정규화에서는 이게 불가능했다.
#
# 실측 근거:
#   - 기존 EGPG 곱셈은 로그분산의 91%가 에너지 -> peakedness 가 사실상 무효
#   - 기존 max 정규화는 실효율 0.017 (명목 0.1) 이며 학습 중 표류
#   자세한 내용: sdrop-paper/00_ACTIVE_SpringerML/수식개정안.md
#
# 6런 x 약 26분 = 약 2.6시간.
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

BASE="--dataset cifar100_lt --imb_ratio 100 --epochs 200 --seed 0 \
      --method sdrop --drop_rate 0.1 --layers L4 \
      --peakedness entropy --norm rank"

# ── beta 곡선 (w=0.5 고정) — "선택 기준이 유효한가" ──────────────
# beta=0 이 무작위 대조군이다. 여기서 성능이 오르면 EGPG 순서가 의미 있는 것.
for b in 0 1 2 4; do
  run "cifar100_lt_sdrop_rate0.1_L4_imb100_pkentropy_nmrank_b${b}_mix0.5_seed0" \
      $BASE --beta $b --mix 0.5
done

# ── w 곡선 (beta=2 고정) — "무엇으로 골라야 하는가" ─────────────
# w=0 은 에너지 단독, w=1 은 확산도 단독. 위 스윕의 w=0.5 와 함께 3점.
for w in 0 1; do
  run "cifar100_lt_sdrop_rate0.1_L4_imb100_pkentropy_nmrank_b2_mix${w}_seed0" \
      $BASE --beta 2 --mix $w
done

echo
echo "##################### 종료 $(date '+%F %T') #####################"
echo "판정은 Few 그룹으로:"
echo "  python eval_tailgroups.py --pattern '*lt*_best.pth' --csv tailgroups_lt.csv"
echo "  python report.py"
echo
echo "beta 곡선이 우상향이면 EGPG 순서가 유효하다는 직접 증거다."
echo "평평하면 선택 기준은 무의미하고 이득의 정체는 약한 정규화다."
