#!/usr/bin/env bash
# Mass-matched random control on Waterbirds (2026-10-07): the Energy-score mask's removed squared-activation mass is matched by random channels, 6 seeds, dumps.
source "$(dirname "$0")/_queue_lib.sh"
W="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir ./checkpoints_final"
for s in 0 1 2 3 4 5; do run "waterbirds_sdrop_energy_massmatch_rate0.1_L4_ep30_nmmean_seed${s}" $W --seed $s --method sdrop_energy_massmatch --drop_rate 0.1 --layers L4; done
