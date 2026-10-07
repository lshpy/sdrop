#!/usr/bin/env bash
# Waterbirds — subpopulation shift. 지표: test_worst_group (history CSV 열).
# 관례(Sagawa'20 ERM): ImageNet ResNet-50, SGD lr 1e-3, wd 1e-4, bs 64, 증강 없음, ~30ep.
# 팔: none / sdrop_random / sdrop_gap / sdrop_energy / sdrop / sdrop_shuffle × L4(7×7)·L3(14×14) × 시드 0-2
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12"
for s in 0 1 2; do
  run "waterbirds_none_rate0.1_none_ep30_seed${s}"  $C --seed $s --method none
  for L in L4 L3; do
    for m in sdrop sdrop_gap sdrop_random sdrop_energy sdrop_shuffle; do
      run "waterbirds_${m}_rate0.1_${L}_ep30_nmmean_seed${s}"  $C --seed $s --method $m --drop_rate 0.1 --layers $L
    done
  done
done
echo "[run_waterbirds] 큐 종료 $(date '+%F %T')"
