#!/usr/bin/env bash
# RSC baseline on Waterbirds (Huang et al. 2020; reviewer 1): paper default (f=1/3, b=1/3) and budget-matched to top-k (f=0.1, b=1), seeds 0-5.
source "$(dirname "$0")/_queue_lib.sh"
CKPT="${CKPT:-./checkpoints_rsc}"; mkdir -p "$CKPT"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT --method none"
for s in 0 1 2 3 4 5; do
  run "waterbirds_none_rate0.1_none_ep30_rscF0.1B1_seed${s}"           $C --seed $s --rsc_f 0.1    --rsc_b 1.0
  run "waterbirds_none_rate0.1_none_ep30_rscF0.3333B0.3333_seed${s}"   $C --seed $s --rsc_f 0.3333 --rsc_b 0.3333
done
echo "[run_rsc] 큐 종료 $(date '+%F %T')"
