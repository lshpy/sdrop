#!/usr/bin/env bash
# Pre-registered backbone-depth robustness (prereg/PREREG_backbone_2026-09-30.md):
# Waterbirds, ResNet-101 (152 dropped 09-30 addendum), ERM / Energy top-k / Shuffled top-k, seeds 0-5, dumps on every arm.
source "$(dirname "$0")/_queue_lib.sh"
CKPT="${CKPT:-./checkpoints_backbone}"; mkdir -p "$CKPT"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for a in resnet101; do
  for s in 0 1 2 3 4 5; do
    run "waterbirds_none_rate0.1_none_${a}_ep30_seed${s}"                       $C --arch $a --seed $s --method none
    run "waterbirds_sdrop_energy_rate0.1_L4_${a}_ep30_nmmean_seltopk_seed${s}"  $C --arch $a --seed $s --method sdrop_energy  --drop_rate 0.1 --layers L4 --select topk
    run "waterbirds_sdrop_shuffle_rate0.1_L4_${a}_ep30_nmmean_seltopk_seed${s}" $C --arch $a --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4 --select topk
  done
done
echo "[run_backbone] 큐 종료 $(date '+%F %T')"
