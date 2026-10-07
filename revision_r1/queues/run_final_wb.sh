#!/usr/bin/env bash
# Reviewer-defence final batch (2026-09-22): the strongest settings (top-k, p=0.3) with the controls the
# headline setting has — shuffled control and feature dumps for the logit-adjustment frontier.
# 24 Waterbirds runs. All --dump_feats so analyze_revision.py can put every arm on the ERM+LA curve.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"   $C --seed $s --method sdrop_energy  --drop_rate 0.1 --layers L4 --select topk
  run "waterbirds_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"       $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4 --select topk
  run "waterbirds_sdrop_energy_rate0.3_L4_ep30_nmmean_seed${s}"           $C --seed $s --method sdrop_energy  --drop_rate 0.3 --layers L4
  run "waterbirds_sdrop_shuffle_rate0.3_L4_ep30_nmmean_seed${s}"               $C --seed $s --method sdrop_shuffle --drop_rate 0.3 --layers L4
done
echo "[run_final_wb] 큐 종료 $(date '+%F %T')"
