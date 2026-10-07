#!/usr/bin/env bash
# Neurocomputing revision add-on (2026-10-06): Waterbirds top-k arm and its shuffled control, seeds 6-13, to bring the
# only arm that clears the logit-adjustment bar from 6 exploratory seeds to 14 paired seeds (same recipe as run_final_wb.sh).
# 16 runs, --dump_feats for the ERM+LA frontier (analyze_revision.py). Output checkpoints_final/.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for s in 6 7 8 9 10 11 12 13; do
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"   $C --seed $s --method sdrop_energy  --drop_rate 0.1 --layers L4 --select topk
  run "waterbirds_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"  $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4 --select topk
done
