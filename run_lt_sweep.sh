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

# ── 가장 중요한 대조군: 실효 드롭률을 맞춘 무작위 선택 ──────────────
#
# "어느 채널을 고르는가"가 정말 중요한지 가리는 유일한 방법은, 같은 양을
# 무작위로 지운 것과 비교하는 것이다. 지금 표의 SpatialDropout 은 p=0.100
# 인데 SDrop_Energy 는 p̄=0.017 이라 강도가 6배 다르다.
#
# 앞서 mean 정규화로 SDrop 을 0.100 까지 "올려" 맞추려 했으나, 그러면 최고
# 채널이 0.59 확률로 드롭되어 파괴적 영역에 들어간다. 반대로 하면 된다 —
# SpatialDropout 을 SDrop 의 0.017 까지 "내려서" 맞춘다. 이쪽은 어떤 채널도
# 파괴적 영역에 들어가지 않는다.
#
#   같은 이득  -> 선택 기준은 무의미. 이득의 정체는 약한 정규화
#   SDrop 우위 -> EGPG 점수가 실제로 옳은 채널을 고르고 있다
#
# sdrop_random(s_c=1) 은 max 정규화에서 모든 채널이 동일 확률이 되므로
# SpatialDropout 과 같은 것이 되어 따로 돌릴 필요가 없다.
run "cifar100_lt_dropout_rate0.017_L4_imb100_seed0" \
    $LT --method dropout --drop_rate 0.017 --layers L4

# max 정규화: 명목값을 올려 실효 드롭률을 끌어올린다
for p in 0.2 0.3 0.5; do
  run "cifar100_lt_sdrop_energy_rate${p}_L4_imb100_seed0" \
      $LT --method sdrop_energy --drop_rate $p --layers L4
done

# mean 정규화: 실효 = 명목이지만, 그게 곧 공정한 비교는 아니다.
#
# max 정규화의 실측 p̄ = 0.0169 에서 점수 분포의 max/mean = 5.9 가 역산된다.
# mean 정규화는 p_drop,c = drop_rate * s_c/mean(s) 이므로 최고점수 채널의
# 드롭 확률이 drop_rate * 5.9 가 된다:
#
#     drop_rate 0.02 -> 0.12      0.05 -> 0.30
#     drop_rate 0.03 -> 0.18      0.10 -> 0.59   <- run_queue.sh 가 돌린 조건
#
# 즉 nmmean 0.1 은 최고 채널을 매 3회 중 2회 가까이 지운다. 논문 §negative_case
# 가 "p_base 가 0.5 에 가까워지면 해당 채널은 학습에 필요한 그래디언트를 아예
# 못 받는다"고 서술한 바로 그 영역이다. 평균을 맞췄다고 개입의 *모양*까지 맞춘
# 것은 아니다 — 같은 평균을 소수 채널에 몰아준 것이다.
#
# 따라서 최고 채널이 안전 영역(0.1~0.2)에 오도록 낮은 쪽을 훑는다. 이때도
# p̄ 는 max 정규화의 0.017 보다 높으므로 "더 세게 개입하면서 tail 을 지키는가"
# 라는 질문에 답이 된다.
for p in 0.02 0.03 0.05; do
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
