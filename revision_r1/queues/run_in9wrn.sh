#!/usr/bin/env bash
# Confirmation (prereg/PREREG_in9bg_wrn_2026-09-30.md): in9bg x1.0, Wide-ResNet-101-2, ERM/top-k/Shuffle top-k, NEW seeds 3-8.
source "$(dirname "$0")/_queue_lib.sh"
CKPT="${CKPT:-./checkpoints_in9wrn}"; mkdir -p "$CKPT"
A=wide_resnet101_2
C="--dataset in9bg --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT --arch $A"
for s in 3 4 5 6 7 8; do
  run "in9bg_none_rate0.1_none_${A}_ep30_seed${s}"                       $C --seed $s --method none
  run "in9bg_sdrop_energy_rate0.1_L4_${A}_ep30_nmmean_seltopk_seed${s}"  $C --seed $s --method sdrop_energy  --drop_rate 0.1 --layers L4 --select topk
  run "in9bg_sdrop_shuffle_rate0.1_L4_${A}_ep30_nmmean_seltopk_seed${s}" $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4 --select topk
done
echo "[run_in9wrn] 큐 종료 $(date '+%F %T')"
