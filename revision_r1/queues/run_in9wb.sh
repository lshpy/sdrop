#!/usr/bin/env bash
# prereg/PREREG_in9wb_2026-10-01.md: in9wb (dog/bird x vehicle-bg/fish-bg, rho 0.95), ResNet-50, ERM/top-k/Shuffle top-k, seeds 0-5.
source "$(dirname "$0")/_queue_lib.sh"
CKPT="${CKPT:-./checkpoints_in9wb}"; mkdir -p "$CKPT"
C="--dataset in9wb --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "in9wb_none_rate0.1_none_ep30_seed${s}"                       $C --seed $s --method none
  run "in9wb_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"  $C --seed $s --method sdrop_energy  --drop_rate 0.1 --layers L4 --select topk
  run "in9wb_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seltopk_seed${s}" $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4 --select topk
done
echo "[run_in9wb] 큐 종료 $(date '+%F %T')"
