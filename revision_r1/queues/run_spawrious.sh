#!/usr/bin/env bash
# Spawrious O2O easy/hard: ERM / Energy / Random, 3 seeds, dumps for LA/AFR (2026-09-16).
source "$(dirname "$0")/_queue_lib.sh"
for ch in o2o_easy o2o_hard; do
  C="--dataset spawrious_${ch} --epochs 10 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
  for s in 0 1 2; do
    run "spawrious_${ch}_none_rate0.1_none_ep10_seed${s}"                 $C --seed $s --method none
    run "spawrious_${ch}_sdrop_energy_rate0.1_L4_ep10_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
    run "spawrious_${ch}_sdrop_random_rate0.1_L4_ep10_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  done
done
echo "[run_spawrious] 큐 종료 $(date '+%F %T')"
