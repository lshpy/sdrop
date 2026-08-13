#!/usr/bin/env bash
# SDrop 실험 큐 — Springer ML 확장판
#
#   bash run_queue.sh            # 전체
#   bash run_queue.sh longtail   # 롱테일만
#   bash run_queue.sh ablation   # 아블레이션만
#   bash run_queue.sh vit        # ViT만
#   bash run_queue.sh grad       # gradient-guided만
#
# 이미 끝난 런은 자동으로 건너뛰므로, 중단 후 다시 실행하면 이어서 진행됩니다.
set -u
GROUP="${1:-all}"
CKPT=./checkpoints
mkdir -p "$CKPT" logs

run () {  # run <run_id> <train.py 인자...>
  local id="$1"; shift
  if [ -f "$CKPT/${id}_history.csv" ]; then
    echo "[건너뜀] $id"
    return
  fi
  echo "──────────────────────────────────────────────────────────"
  echo "[실행] $id      $(date '+%F %T')"
  python train.py "$@" 2>&1 | tee "logs/${id}.log"
  echo "[완료] $id      $(date '+%F %T')"
}

C="--dataset cifar100 --epochs 200"
LT="--dataset cifar100_lt --imb_ratio 100 --epochs 200"
VIT="--epochs 200 --lr 1e-3 --strong_aug --mixup_alpha 0.2 --warmup_epochs 10 --label_smoothing 0.1"

# ── 롱테일: 가설의 정면 검증. 가장 중요하므로 맨 앞. ───────────────
if [[ "$GROUP" == all || "$GROUP" == longtail ]]; then
  for s in 0 1 2; do
    for m in none dropout sdrop_energy sdrop; do
      run "cifar100_lt_${m}_rate0.1_L4_imb100_seed${s}" \
          $LT --method $m --drop_rate 0.1 --layers L4 --seed $s
    done
    # 평균보존 정규화 — 실효 드롭률을 명목값과 일치시킨 조건
    run "cifar100_lt_sdrop_energy_rate0.1_L4_imb100_nmmean_seed${s}" \
        $LT --method sdrop_energy --drop_rate 0.1 --layers L4 --norm mean --seed $s
  done
fi

# ── 아블레이션: 표의 빈칸 채우기 ──────────────────────────────────
if [[ "$GROUP" == all || "$GROUP" == ablation ]]; then
  for s in 0 1 2; do
    run "cifar100_sdrop_random_rate0.1_L4_seed${s}" \
        $C --method sdrop_random --drop_rate 0.1 --layers L4 --seed $s
    run "cifar100_sdrop_peak_rate0.1_L4_seed${s}" \
        $C --method sdrop_peak --drop_rate 0.1 --layers L4 --seed $s
    run "cifar100_dropout_std_rate0.1_L4_seed${s}" \
        $C --method dropout_std --drop_rate 0.1 --layers L4 --seed $s
    run "cifar100_sgridlc_rate0.3_L3_G4_seed${s}" \
        $C --method sgridlc --drop_rate 0.3 --layers L3 --grid_size 4 --seed $s
  done
  for s in 1 2; do
    run "cifar100_dropblock_rate0.1_L4_seed${s}" $C --method dropblock --drop_rate 0.1 --layers L4 --seed $s
    run "cifar100_senet_rate0.1_L4_seed${s}"     $C --method senet     --drop_rate 0.1 --layers L4 --seed $s
    run "cifar100_cbam_rate0.1_L4_seed${s}"      $C --method cbam      --drop_rate 0.1 --layers L4 --seed $s
  done
fi

# ── gradient-guided: suppress vs amplify 판별 ────────────────────
if [[ "$GROUP" == all || "$GROUP" == grad ]]; then
  for s in 0 1 2; do
    run "cifar100_sdrop_rate0.1_L4_gradsuppress_seed${s}" \
        $C --method sdrop --drop_rate 0.1 --layers L4 --grad_mode suppress --seed $s
  done
  run "cifar100_sdrop_rate0.1_L4_gradamplify_seed0" \
      $C --method sdrop --drop_rate 0.1 --layers L4 --grad_mode amplify --seed 0
  # 롱테일 + grad-guided 조합
  for s in 0 1 2; do
    run "cifar100_lt_sdrop_rate0.1_L4_imb100_gradsuppress_seed${s}" \
        $LT --method sdrop --drop_rate 0.1 --layers L4 --grad_mode suppress --seed $s
  done
fi

# ── ViT: 시드 확장 + 무작위 DropHead 대조군 ──────────────────────
if [[ "$GROUP" == all || "$GROUP" == vit ]]; then
  for s in 3 4; do
    run "cifar100_vit_rate0.1_none_seed${s}"            $C --method vit $VIT --seed $s
    run "cifar100_sdrop_vit_rate0.3_L3+L4_seed${s}"     $C --method sdrop_vit --drop_rate 0.3 --layers L3 L4 $VIT --seed $s
  done
fi

# ── 드롭률 스윕 (100에폭 단축) ───────────────────────────────────
if [[ "$GROUP" == all || "$GROUP" == sweep ]]; then
  for p in 0.05 0.2 0.3 0.5; do
    run "cifar100_sdrop_energy_rate${p}_L4_ep100_seed0" \
        --dataset cifar100 --epochs 100 --method sdrop_energy --drop_rate $p --layers L4 --seed 0
  done
fi

echo
echo "##################### 큐 종료 $(date '+%F %T') #####################"
ls -1 "$CKPT"/*history.csv 2>/dev/null | wc -l | xargs echo "완료된 런:"
python summarize.py 2>/dev/null || true
