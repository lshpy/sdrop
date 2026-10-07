#!/usr/bin/env bash
# Revision (2026-09-15): CelebA to 6 seeds (none / gap / energy), dumps, + Spectral Decoupling baseline.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset celeba --epochs 8 --lr 0.001 --weight_decay 1e-4 --batch_size 128 --norm mean --num_workers 14 --dump_feats --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "celeba_none_rate0.1_none_ep8_seed${s}"                      $C --seed $s --method none
  run "celeba_sdrop_gap_rate0.1_L4_ep8_nmmean_seed${s}"            $C --seed $s --method sdrop_gap --drop_rate 0.1 --layers L4
  run "celeba_none_rate0.1_none_ep8_sd0.1_seed${s}"                $C --seed $s --method none --sd_lambda 0.1
done
for s in 3 4 5; do
  run "celeba_sdrop_energy_rate0.1_L4_ep8_nmmean_seed${s}"         $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
done
echo "[run_revision_celeba] 큐 종료 $(date '+%F %T')"
