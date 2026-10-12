#!/usr/bin/env bash
# 2026-08-28 파드 삭제로 소실된 런 재실행 — Springer ML 표의 빈칸 전부.
#
#   SHARD=$i NSHARD=4 CUDA_VISIBLE_DEVICES=$i bash run_missing.sh
#
# 주의 1 (에폭): 드롭률/삽입위치 스윕은 기존 기준점
#   cifar100_sdrop_energy_rate0.1_L4_* 가 200에폭이므로 여기서도 200으로 돈다.
#   run_queue.sh 의 sweep 그룹은 100에폭이라 비교가 불가능했다.
# 주의 2 (ID): train.py 는 sdrop_* 계열에 항상 _nm{norm} 을 붙인다(기본 mean).
#   기존 무태그 파일은 legacy 'max' 런이므로, 비교 대상이 되려면 --norm max 로 돌려
#   _nmmax 태그가 붙어야 한다. summarize_for_paper.py 도 같이 고쳐 두었다.
set -u
SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"
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
  echo "[완료] $id  $(date '+%F %T')"
}

C="--dataset cifar100 --epochs 200"
VIT="--epochs 200 --lr 1e-3 --strong_aug --mixup_alpha 0.2 --warmup_epochs 10 --label_smoothing 0.1"
MAX="--norm max"   # legacy 무태그 런과 같은 의미 → _nmmax 로 저장됨

# ── 드롭률 민감도 (200ep, seed 0) ────────────────────────────────
for p in 0.05 0.2 0.3 0.5; do
  run "cifar100_sdrop_energy_rate${p}_L4_nmmax_seed0" \
      $C --method sdrop_energy --drop_rate $p --layers L4 $MAX --seed 0
done

# ── 삽입 위치 (200ep, seed 0) ────────────────────────────────────
run "cifar100_sdrop_energy_rate0.1_L3_nmmax_seed0" \
    $C --method sdrop_energy --drop_rate 0.1 --layers L3 $MAX --seed 0
run "cifar100_sdrop_energy_rate0.1_L3+L4_nmmax_seed0" \
    $C --method sdrop_energy --drop_rate 0.1 --layers L3 L4 $MAX --seed 0

# ── 점수 함수 옵션 (200ep, seed 0) ───────────────────────────────
run "cifar100_sdrop_rate0.1_L4_nmmean_seed0" \
    $C --method sdrop --drop_rate 0.1 --layers L4 --norm mean --seed 0
run "cifar100_sdrop_rate0.1_L4_pkentropy_nmmax_seed0" \
    $C --method sdrop --drop_rate 0.1 --layers L4 --peakedness entropy $MAX --seed 0
run "cifar100_sdrop_rate0.1_L4_nmmax_sg_seed0" \
    $C --method sdrop --drop_rate 0.1 --layers L4 $MAX --self_gamma --seed 0

# ── ViT 시드 확장 + 삽입 위치 + 점수 입력 ────────────────────────
for s in 3 4; do
  run "cifar100_vit_rate0.1_none_seed${s}" $C --method vit $VIT --seed $s
  run "cifar100_sdrop_vit_rate0.3_L3+L4_nmmax_seed${s}" \
      $C --method sdrop_vit --drop_rate 0.3 --layers L3 L4 $MAX $VIT --seed $s
done
run "cifar100_sdrop_vit_rate0.3_L3+L4_nmmax_scorepost_seed0" \
    $C --method sdrop_vit --drop_rate 0.3 --layers L3 L4 $MAX --vit_score post $VIT --seed 0
run "cifar100_sdrop_vit_rate0.3_L3_nmmax_seed0" \
    $C --method sdrop_vit --drop_rate 0.3 --layers L3 $MAX $VIT --seed 0
run "cifar100_sdrop_vit_full_rate0.3_none_nmmax_seed0" \
    $C --method sdrop_vit_full --drop_rate 0.3 $MAX $VIT --seed 0

# ── 시드가 모자란 격자 칸 메우기 (100ep — 격자와 동일) ───────────
G="--dataset cifar100_lt --imb_ratio 100 --drop_rate 0.1 --epochs 100"
run "cifar100_lt_sdrop_energy_rate0.1_L4_imb100_nmmean_w0.25_seed2" \
    $G --method sdrop_energy --layers L4 --norm mean --width_mult 0.25 --seed 2
run "cifar100_lt_sdrop_random_rate0.1_L4_imb100_nmmean_w0.25_seed2" \
    $G --method sdrop_random --layers L4 --norm mean --width_mult 0.25 --seed 2
run "cifar100_lt_dropout_rate0.1_L4_imb100_seed2" \
    --dataset cifar100_lt --imb_ratio 100 --epochs 200 \
    --method dropout --drop_rate 0.1 --layers L4 --seed 2

echo "=== SHARD=$SHARD 완료 $(date '+%F %T') ==="
