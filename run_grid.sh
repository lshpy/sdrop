#!/usr/bin/env bash
# 채널 희소성 격자 — 독점 가설의 정면 검증.
#
# 가설: SDrop의 효과는 채널이 희소할수록(클래스는 많고 채널은 적을수록) 커진다.
# 두 축으로 희소성을 만든다:
#   축 A  클래스 불균형   --imb_ratio 1 / 10 / 50 / 100 / 200
#   축 B  네트워크 폭     --width_mult 1.0 / 0.5 / 0.25
#
#   bash run_grid.sh imb      # 축 A (폭 고정 1.0)
#   bash run_grid.sh width    # 축 B (불균형 고정 100)
#   bash run_grid.sh all
#
# 샤딩은 run_rented.sh 와 동일:
#   SHARD=$i NSHARD=8 CUDA_VISIBLE_DEVICES=$g bash run_grid.sh all
set -u
GROUP="${1:-all}"
SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"
SEEDS="${SEEDS:-0 1 2}"
# 격자는 조건 간 *비교*가 목적이므로 200 대신 100에폭으로 돈다.
# (논문에 "condition sweep run at reduced epochs"로 명시할 것)
EPOCHS="${EPOCHS:-100}"
CKPT=./checkpoints
mkdir -p "$CKPT" logs
AMP_FLAG=""; [ "${AMP:-1}" = "1" ] && AMP_FLAG="--amp"

IDX=0
run () {
  local id="$1"; shift
  local mine=$(( IDX % NSHARD )); IDX=$(( IDX + 1 ))
  [ "$mine" != "$SHARD" ] && return
  if [ -f "$CKPT/${id}_history.csv" ]; then echo "[건너뜀] $id"; return; fi
  echo "[실행] $id  $(date '+%F %T')"
  python train.py "$@" $AMP_FLAG 2>&1 | tee "logs/${id}.log"
}
want () { [[ "$GROUP" == all || "$GROUP" == "$1" ]]; }

# 세 방법을 나란히: 대조군(random) / 기존 점수(EGPG) / 신규 점수(class-aware)
# baseline(none)은 각 조건마다 반드시 필요하다 -- 차이를 재는 기준선이므로.
METHODS="none sdrop_random sdrop_energy sdrop_class"

cell () {   # cell <dataset-args...> -- <id-suffix>
  local dsargs="$1" idsuf="$2" extra="${3:-}"
  for m in $METHODS; do
    for s in $SEEDS; do
      local layers="--layers L4"; [ "$m" = none ] && layers=""
      local lstr="L4";            [ "$m" = none ] && lstr="none"
      local nm=""; case "$m" in sdrop_*) nm="_nmmean";; esac
      run "cifar100${idsuf}_${m}_rate0.1_${lstr}${extra}${nm}_seed${s}" \
          $dsargs --method "$m" --drop_rate 0.1 $layers --epochs "$EPOCHS" --seed "$s"
    done
  done
}

# ── 축 A: 불균형 (폭 1.0) ────────────────────────────────────────────
if want imb; then
  cell "--dataset cifar100"                        ""            ""
  for r in 10 50 100 200; do
    cell "--dataset cifar100_lt --imb_ratio $r"    "_lt"         "_imb${r}"
  done
fi

# ── 축 B: 폭 (불균형 100 고정) ───────────────────────────────────────
if want width; then
  for w in 0.5 0.25; do
    for m in $METHODS; do
      for s in $SEEDS; do
        layers="--layers L4"; lstr="L4"
        [ "$m" = none ] && { layers=""; lstr="none"; }
        nm=""; case "$m" in sdrop_*) nm="_nmmean";; esac
        run "cifar100_lt_${m}_rate0.1_${lstr}_imb100${nm}_w${w}_seed${s}" \
            --dataset cifar100_lt --imb_ratio 100 --method "$m" \
            --drop_rate 0.1 $layers --width_mult "$w" --epochs "$EPOCHS" --seed "$s"
      done
    done
  done
fi

echo "=== GROUP=$GROUP SHARD=$SHARD 완료 $(date '+%F %T') ==="
