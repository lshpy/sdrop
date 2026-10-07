#!/usr/bin/env bash
# Colored-MNIST strength sweep (LfF protocol) + BAR (2026-09-17). Dumps for conflict-accuracy / LA analysis.
source "$(dirname "$0")/_queue_lib.sh"
for b in 0.95 0.98 0.995; do
  C="--dataset cmnist_${b} --epochs 20 --lr 0.1 --weight_decay 5e-4 --batch_size 128 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
  for s in 0 1 2; do
    run "cmnist_${b}_none_rate0.1_none_ep20_seed${s}"                 $C --seed $s --method none
    run "cmnist_${b}_sdrop_energy_rate0.1_L4_ep20_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
    run "cmnist_${b}_sdrop_random_rate0.1_L4_ep20_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  done
done
B="--dataset bar --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "bar_none_rate0.1_none_ep30_seed${s}"                          $B --seed $s --method none
  run "bar_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"             $B --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  run "bar_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"             $B --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  run "bar_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seed${s}"            $B --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4
done
echo "[run_cmnist_bar] 큐 종료 $(date '+%F %T')"
