#!/usr/bin/env bash
# CelebA (Blond × Male) — Waterbirds 와 같은 팔·L4. 8ep, lr 1e-3, wd 1e-4, bs 128.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset celeba --epochs 8 --lr 0.001 --weight_decay 1e-4 --batch_size 128 --norm mean --num_workers 14"
for s in 0 1 2; do
  run "celeba_none_rate0.1_none_ep8_seed${s}"                 $C --seed $s --method none
  for m in sdrop_energy sdrop_gap sdrop_random sdrop_shuffle; do
    run "celeba_${m}_rate0.1_L4_ep8_nmmean_seed${s}"           $C --seed $s --method $m --drop_rate 0.1 --layers L4
  done
done
echo "[run_celeba] 큐 종료 $(date '+%F %T')"
