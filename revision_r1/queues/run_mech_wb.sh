#!/usr/bin/env bash
# Mechanism checks on Waterbirds (2026-09-17): reverse control, early/late-only schedules, drop-rate dose-response.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "waterbirds_sdrop_reverse_rate0.1_L4_ep30_nmmean_seed${s}"            $C --seed $s --method sdrop_reverse --drop_rate 0.1 --layers L4
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_schearly_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4 --schedule early
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_wu0.5_seed${s}"       $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4 --warmup_frac 0.5
done
for r in 0.05 0.2 0.3; do for s in 0 1 2 3 4 5; do
  run "waterbirds_sdrop_energy_rate${r}_L4_ep30_nmmean_seed${s}"            $C --seed $s --method sdrop_energy --drop_rate $r --layers L4
done; done
for s in 0 1 2 3 4 5; do
  run "waterbirds_sdrop_random_rate0.3_L4_ep30_nmmean_seed${s}"             $C --seed $s --method sdrop_random --drop_rate 0.3 --layers L4
done
echo "[run_mech_wb] 큐 종료 $(date '+%F %T')"
