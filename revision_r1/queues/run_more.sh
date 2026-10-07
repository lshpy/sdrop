#!/usr/bin/env bash
# More datasets / seeds (2026-09-22): Spawrious O2O-Medium (new), O2O-Easy seeds 3-5 (to six), CelebA Random/Shuffled seeds 3-5 (to six).
source "$(dirname "$0")/_queue_lib.sh"
for ch in o2o_medium o2o_easy; do
  C="--dataset spawrious_${ch} --epochs 10 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
  [ $ch = o2o_medium ] && seeds="0 1 2" || seeds="3 4 5"
  for s in $seeds; do
    run "spawrious_${ch}_none_rate0.1_none_ep10_seed${s}"                 $C --seed $s --method none
    run "spawrious_${ch}_sdrop_energy_rate0.1_L4_ep10_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
    run "spawrious_${ch}_sdrop_random_rate0.1_L4_ep10_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  done
done
CC="--dataset celeba --epochs 8 --lr 0.001 --weight_decay 1e-4 --batch_size 128 --norm mean --num_workers 8 --save_dir $CKPT"
for s in 3 4 5; do
  run "celeba_sdrop_random_rate0.1_L4_ep8_nmmean_seed${s}"     $CC --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  run "celeba_sdrop_shuffle_rate0.1_L4_ep8_nmmean_seed${s}"    $CC --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4
done
echo "[run_more] 큐 종료 $(date '+%F %T')"
