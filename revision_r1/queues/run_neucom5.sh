#!/usr/bin/env bash
# ERM (no dropout) Waterbirds seeds 10-13 with feature dumps, so the top-k arm's logit-adjustment frontier is paired on all 14 seeds.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for s in 10 11 12 13; do run "waterbirds_none_rate0.1_none_ep30_seed${s}" $C --seed $s --method none; done
