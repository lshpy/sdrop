#!/usr/bin/env bash
# CIFAR-100-LT 드롭률 스윕 — 시드 0 에서만 돈다.
#
#   bash run_lt_sweep.sh
#
# 왜 시드 0 만인가:
#   보고에 쓰는 10,000장은 validation 이자 test 다. 여기서 여러 설정을 돌려보고
#   가장 높은 것을 고르면 그 숫자는 더 이상 일반화 성능이 아니다. LT 학습셋은
#   tail 클래스가 5장뿐이라 별도 val 을 떼어낼 수도 없다(떼면 1장 남는다).
#   그래서 차선책을 쓴다 — 설정 탐색은 시드 0 에서만 하고, 고른 설정을
#   시드 1·2 에서 다시 돌려 그 두 개를 보고한다. 시드 1·2 는 탐색에 쓰이지
#   않았으므로 held-out 이다.
#
# 왜 이 스윕이 필요한가:
#   측정된 실효 드롭률이 명목 0.1 에서 p̄ = 0.017 이다 (5.9배 약함). 즉 LT 에서
#   실제로 시험된 개입 강도는 1.7% 하나뿐이고, 그보다 센 영역은 미탐색이다.
#   run_queue.sh 의 sweep 그룹은 균형 CIFAR-100 에서만 돈다.
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

LT="--dataset cifar100_lt --imb_ratio 100 --epochs 200 --seed 0"

# max 정규화: 명목값을 올려 실효 드롭률을 끌어올린다
for p in 0.2 0.3 0.5; do
  run "cifar100_lt_sdrop_energy_rate${p}_L4_imb100_seed0" \
      $LT --method sdrop_energy --drop_rate $p --layers L4
done

# mean 정규화: 실효 = 명목. 0.1 은 run_queue.sh 가 이미 돌리므로 그 위쪽만.
for p in 0.05 0.2; do
  run "cifar100_lt_sdrop_energy_rate${p}_L4_imb100_nmmean_seed0" \
      $LT --method sdrop_energy --drop_rate $p --layers L4 --norm mean
done

# L3+L4 동시 삽입 — 지금까지 L4 단독만 시험했다
run "cifar100_lt_sdrop_energy_rate0.1_L3+L4_imb100_seed0" \
    $LT --method sdrop_energy --drop_rate 0.1 --layers L3 L4

echo
echo "##################### 스윕 종료 $(date '+%F %T') #####################"
echo "그룹별 정확도로 판정하세요 (전체 정확도 아님):"
echo "  python eval_tailgroups.py --pattern '*lt*seed0_best.pth'"
echo
echo "Few 그룹이 가장 높은 설정을 고른 뒤, 그 설정만 시드 1·2 로 다시 돌리세요."
echo "보고에는 시드 1·2 를 씁니다 — 시드 0 은 설정 선택에 이미 소모됐습니다."
