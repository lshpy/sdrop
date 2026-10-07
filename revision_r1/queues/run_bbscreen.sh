#!/usr/bin/env bash
# Backbone screen stage 1 (prereg/PLAN_backbone_screen_2026-09-30.md): Wide-ResNet-101-2 x {waterbirds, coco_catdog, in9bg_s05, counteranimal, in9bg}
# x {ERM, Energy top-k, Shuffle top-k} x seeds 0-2 = 45 runs. SHARD/NSHARD split across pods.
source "$(dirname "$0")/_queue_lib.sh"
CKPT="${CKPT:-./checkpoints_bbscreen}"; mkdir -p "$CKPT"
A=wide_resnet101_2
for ds in waterbirds coco_catdog in9bg_s05 counteranimal in9bg; do
  C="--dataset $ds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT --arch $A"
  for s in 0 1 2; do
    run "${ds}_none_rate0.1_none_${A}_ep30_seed${s}"                       $C --seed $s --method none
    run "${ds}_sdrop_energy_rate0.1_L4_${A}_ep30_nmmean_seltopk_seed${s}"  $C --seed $s --method sdrop_energy  --drop_rate 0.1 --layers L4 --select topk
    run "${ds}_sdrop_shuffle_rate0.1_L4_${A}_ep30_nmmean_seltopk_seed${s}" $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4 --select topk
  done
done
echo "[run_bbscreen] 큐 종료 $(date '+%F %T')"
