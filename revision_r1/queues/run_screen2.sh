#!/usr/bin/env bash
# NICO++ HARD pairs screen (2026-09-23): visually confusable classes, grass vs water shortcut, 95% train / balanced 150-per-cell test.
# Cells (grass, water): lion 460/253, kangaroo 346/252 | sheep 438/292, kangaroo 346/252 | umbrella 303/364, tent 389/441.
# <5k train images -> 30 epochs. ERM seeds 0-2, Energy/Random seeds 0-5 (no shuffle), all with dumps.
source "$(dirname "$0")/_queue_lib.sh"
for pair in "lion,kangaroo" "sheep,kangaroo" "umbrella,tent"; do
  ds="nicopp:${pair}:grass,water"; tag="nicopp_${pair/,/-}_grass-water"     # run_id(): ',' -> '-', ':' -> '_'
  C="--dataset $ds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
  for s in 0 1 2; do
    run "${tag}_none_rate0.1_none_ep30_seed${s}"                 $C --seed $s --method none
  done
  for s in 0 1 2 3 4 5; do
    run "${tag}_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
    run "${tag}_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  done
done
echo "[run_screen2] 큐 종료 $(date '+%F %T')"
