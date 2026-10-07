#!/usr/bin/env bash
# JTT baseline (Liu et al. ICML'21) on Waterbirds: ERM T epochs -> training error set x lambda -> retrain from scratch. 6 seeds x 2 configs = 12.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "waterbirds_none_rate0.1_none_ep30_jttT2L50_seed${s}"  $C --seed $s --method none --jtt_stage1_epochs 2 --jtt_lambda 50
  run "waterbirds_none_rate0.1_none_ep30_jttT5L20_seed${s}"  $C --seed $s --method none --jtt_stage1_epochs 5 --jtt_lambda 20
done
echo "[run_jtt_wb] 큐 종료 $(date '+%F %T')"
