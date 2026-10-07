#!/usr/bin/env bash
# Screen for a second positive background-shortcut benchmark (2026-09-23): NICO++ elephant/sailboat x grass/water.
# 558 train images (<5k) -> 30 epochs. ERM seeds 0-2, Energy/Random/Shuffle seeds 0-5, all with dumps for LA/AFR analysis.
source "$(dirname "$0")/_queue_lib.sh"
for ds in nicopp; do
  C="--dataset $ds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
  for s in 0 1 2; do
    run "${ds}_none_rate0.1_none_ep30_seed${s}"                 $C --seed $s --method none
  done
  for s in 0 1 2 3 4 5; do
    run "${ds}_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_energy  --drop_rate 0.1 --layers L4
    run "${ds}_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_random  --drop_rate 0.1 --layers L4
    run "${ds}_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seed${s}"   $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4
  done
done
echo "[run_screen] 큐 종료 $(date '+%F %T')"
