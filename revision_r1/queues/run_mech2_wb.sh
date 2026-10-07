#!/usr/bin/env bash
# Mechanism batch 2 (2026-09-23): per-channel gradient logging on the headline arms + cross-sample control.
# 15 Waterbirds runs: none/energy/random x seeds 0-2 (with --log_grad_channels), sdrop_cross x seeds 0-5.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --log_grad_channels --save_dir $CKPT"
for s in 0 1 2; do
  run "waterbirds_none_rate0.1_none_ep30_seed${s}"                 $C --seed $s --method none
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  run "waterbirds_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
done
for s in 0 1 2 3 4 5; do
  run "waterbirds_sdrop_cross_rate0.1_L4_ep30_nmmean_seed${s}"     $C --seed $s --method sdrop_cross --drop_rate 0.1 --layers L4
done
echo "[run_mech2_wb] 큐 종료 $(date '+%F %T')"
