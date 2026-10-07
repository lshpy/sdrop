#!/usr/bin/env bash
# Waterbirds with a randomly initialised ResNet-50 (2026-09-22): does the effect need the pretrained dominant channels?
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.01 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --no_pretrained --dump_feats --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "waterbirds_none_rate0.1_none_ep30_scratch_seed${s}"                 $C --seed $s --method none
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_scratch_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  run "waterbirds_sdrop_random_rate0.1_L4_ep30_scratch_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
done
echo "[run_scratch_wb] 큐 종료 $(date '+%F %T')"
