#!/usr/bin/env bash
# Mechanism batch 3 (2026-09-23): batch 2 headline arms re-run with the CLEAN per-channel gradient measurement
# (--log_grad_clean 8: 8 fixed train batches, eval() mode -> SDrop identity, no mask confound) next to the
# ordinary --log_grad_channels. 9 Waterbirds runs: none/energy/random x seeds 0-2. Same run ids as batch 2,
# so the pod entry must point CKPT at a fresh dir (checkpoints_mech3).
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --log_grad_channels --log_grad_clean 8 --save_dir $CKPT"
for s in 0 1 2; do
  run "waterbirds_none_rate0.1_none_ep30_seed${s}"                 $C --seed $s --method none
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  run "waterbirds_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
done
echo "[run_mech3_wb] 큐 종료 $(date '+%F %T')"
