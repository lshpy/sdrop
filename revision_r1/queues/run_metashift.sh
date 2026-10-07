#!/usr/bin/env bash
# MetaShift cat-vs-dog (DISC subset, indoor/outdoor background shortcut): ERM / Energy / Random / Shuffle, 6 seeds (2026-09-22).
# 1,011 train images -> 30 epochs. Dumps for LA/AFR analysis.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset metashift --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "metashift_none_rate0.1_none_ep30_seed${s}"                    $C --seed $s --method none
  run "metashift_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"       $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  run "metashift_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"       $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  run "metashift_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seed${s}"      $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4
done
echo "[run_metashift] 큐 종료 $(date '+%F %T')"
