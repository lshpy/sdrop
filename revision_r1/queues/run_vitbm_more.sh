#!/usr/bin/env bash
# ViT-B MLP 뉴런 자리 시드 3-5 보강 (none / GAP / random / shuffle) → 6시드
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --self_gamma --layers L4"
for s in 3 4 5; do
  run "waterbirds_vitb_none_rate0.1_L4_vit_b_16_ep30_seed${s}"                  $C --seed $s --method vitb_none
  for m in vitbm_sdrop_gap vitbm_sdrop_random vitbm_sdrop_shuffle; do
    run "waterbirds_${m}_rate0.1_L4_vit_b_16_ep30_nmmean_sg_seed${s}"          $C --seed $s --method $m --drop_rate 0.1
  done
done
echo "[run_vitbm_more] 큐 종료 $(date '+%F %T')"
