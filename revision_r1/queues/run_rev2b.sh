#!/usr/bin/env bash
# Reviewer-response batch (2026-09-24), Waterbirds, ResNet-50, L4, p_base 0.1 unless stated. Dumps kept, weights deleted after each run.
# A. RWY: does the gain survive once the class prior is removed without shrinking the training set?  (ERM / Energy / Random, 6 seeds)
# B. Seeds 6-24 of the headline three arms, for a CI on the ERM+LA margin and a Holm-correctable family.
# C. ERM weight-decay sweep {1e-3,1e-2,1e-1} x 3 seeds, then Energy at the same weight decays (is SDrop a substitute for regularisation?).
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
W="--weight_decay 1e-4"
clean () { rm -f "$CKPT"/*_best.pth; }
for s in $(seq 6 15); do
  run "waterbirds_none_rate0.1_none_ep30_seed${s}"                      $C $W --seed $s --method none; clean
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"         $C $W --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4; clean
  run "waterbirds_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"         $C $W --seed $s --method sdrop_random --drop_rate 0.1 --layers L4; clean
done
echo "[run_rev2b] 큐 종료 $(date '+%F %T')"
