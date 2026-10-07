#!/usr/bin/env bash
# Max-Drop comparison (Park & Kwak 2016), 2026-10-07: the closest published strong-unit rule, on both regimes.
#   CIFAR-100 10% data, L4, 6 seeds: maxdrop (published feature-wise rule, p=0.5 per position) and maxdrop_ch (budget-matched channel top-k, p=0.1)
#   Waterbirds, L4, 6 seeds, dumps for the LA frontier: same two arms. 24 runs.
source "$(dirname "$0")/_queue_lib.sh"
r () { run "$@"; rm -f "$CKPT/$1_best.pth"; return 0; }
C="--dataset cifar100 --epochs 200 --subset_frac 0.1 --norm mean --num_workers 6 --save_dir ./checkpoints"
for s in 0 1 2 3 4 5; do
  CKPT=./checkpoints r "cifar100_maxdrop_rate0.5_L4_sub0.1_seed${s}"     $C --seed $s --method maxdrop    --drop_rate 0.5 --layers L4
  CKPT=./checkpoints r "cifar100_maxdrop_ch_rate0.1_L4_sub0.1_seed${s}"  $C --seed $s --method maxdrop_ch --drop_rate 0.1 --layers L4
done
W="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir ./checkpoints_final"
for s in 0 1 2 3 4 5; do
  CKPT=./checkpoints_final r "waterbirds_maxdrop_rate0.5_L4_ep30_seed${s}"     $W --seed $s --method maxdrop    --drop_rate 0.5 --layers L4
  CKPT=./checkpoints_final r "waterbirds_maxdrop_ch_rate0.1_L4_ep30_seed${s}"  $W --seed $s --method maxdrop_ch --drop_rate 0.1 --layers L4
done
