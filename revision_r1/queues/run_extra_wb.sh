#!/usr/bin/env bash
# Reviewer-defence extras (2026-09-17): CelebA Energy seeds 0-2 with dumps (6-seed set), CelebA SUBY none,
# Waterbirds deterministic top-k selection.
source "$(dirname "$0")/_queue_lib.sh"
CC="--dataset celeba --epochs 8 --lr 0.001 --weight_decay 1e-4 --batch_size 128 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
for s in 0 1 2; do
  run "celeba_sdrop_energy_rate0.1_L4_ep8_nmmean_seed${s}"      $CC --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
done
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"  $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4 --select topk
done
for s in 0 1 2; do
  run "celeba_none_rate0.1_none_ep8_cb_seed${s}"                $CC --seed $s --method none --class_balance
done
echo "[run_extra_wb] 큐 종료 $(date '+%F %T')"
