#!/usr/bin/env bash
# 롱테일 시드 증량 — run_queue.sh 의 시드 0·1·2 위에 더 쌓는다.
#
#   bash run_lt_seeds.sh 3 4       # 시드 3,4 추가 (총 5시드)
#   bash run_lt_seeds.sh 3 9       # 시드 3..9 추가 (총 10시드)
#
# 왜 필요한가:
#   실측된 시드 짝지은 차이의 표준편차가 Few 그룹에서 약 0.9 pp 다. 3시드면
#   표준오차가 0.52 pp 라 +1.4 pp 는 잡히지만 +0.5 pp 급 차이(SDrop vs
#   SpatialDropout)는 못 가린다. 5시드 -> 0.41, 10시드 -> 0.29 로 내려간다.
#   효과를 키우는 게 아니라 노이즈를 줄이는 접근이라 결과 해석이 깨끗하다.
#
# 런당 약 26분. 5조건 x 시드 1개 = 2.2시간.
set -u
FROM="${1:?사용법: bash run_lt_seeds.sh <시작시드> <끝시드>}"
TO="${2:?사용법: bash run_lt_seeds.sh <시작시드> <끝시드>}"
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

LT="--dataset cifar100_lt --imb_ratio 100 --epochs 200"

for s in $(seq "$FROM" "$TO"); do
  for m in none dropout sdrop_energy sdrop; do
    run "cifar100_lt_${m}_rate0.1_L4_imb100_seed${s}" \
        $LT --method $m --drop_rate 0.1 --layers L4 --seed $s
  done
  run "cifar100_lt_sdrop_energy_rate0.1_L4_imb100_nmmean_seed${s}" \
      $LT --method sdrop_energy --drop_rate 0.1 --layers L4 --norm mean --seed $s
done

echo
echo "##################### 종료 $(date '+%F %T') #####################"
echo "  python eval_tailgroups.py --pattern '*lt*_best.pth' --csv tailgroups_lt.csv"
echo "  python report.py"
