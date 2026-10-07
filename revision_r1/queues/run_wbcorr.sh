#!/usr/bin/env bash
# prereg/PREREG_wbcorr_geometry_2026-10-01.md: Waterbirds ERM at train_corr 0.9/0.75/0.5, seeds 0-2, with feature dumps.
source "$(dirname "$0")/_queue_lib.sh"
CKPT="${CKPT:-./checkpoints_wbcorr}"; mkdir -p "$CKPT"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
for r in 0.9 0.75 0.5; do for s in 0 1 2; do
  run "waterbirds_none_rate0.1_none_ep30_corr${r}_seed${s}" $C --seed $s --method none --train_corr $r
done; done
echo "[run_wbcorr] 큐 종료 $(date '+%F %T')"
