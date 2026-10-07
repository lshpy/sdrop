#!/usr/bin/env bash
# Revision (2026-09-15): Waterbirds 6 seeds, dumps for logit-adjustment / DFR, + Spectral Decoupling baseline.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --dump_feats --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "waterbirds_none_rate0.1_none_ep30_seed${s}"                 $C --seed $s --method none
  run "waterbirds_sdrop_gap_rate0.1_L4_ep30_nmmean_seed${s}"       $C --seed $s --method sdrop_gap --drop_rate 0.1 --layers L4
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  run "waterbirds_none_rate0.1_none_ep30_sd0.1_seed${s}"           $C --seed $s --method none --sd_lambda 0.1
done
echo "[run_revision_wb] 큐 종료 $(date '+%F %T')"
