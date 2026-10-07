#!/usr/bin/env bash
# Spatial-statistics dump (2026-09-23): ERM seed 0 on four datasets, first 512 test images -> <id>_spatial.npz.
# Per-dataset hyperparameters as in run_waterbirds/run_celeba/run_metashift/run_spawrious; --dump_spatial adds the _sp tag.
source "$(dirname "$0")/_queue_lib.sh"
C="--lr 0.001 --weight_decay 1e-4 --norm mean --num_workers 6 --seed 0 --method none --dump_feats --dump_spatial 512 --save_dir $CKPT"
run "waterbirds_none_rate0.1_none_ep30_sp_seed0"          $C --dataset waterbirds         --epochs 30 --batch_size 64
run "celeba_none_rate0.1_none_ep8_sp_seed0"               $C --dataset celeba             --epochs 8  --batch_size 128
run "metashift_none_rate0.1_none_ep30_sp_seed0"           $C --dataset metashift          --epochs 30 --batch_size 64
run "spawrious_o2o_easy_none_rate0.1_none_ep10_sp_seed0"  $C --dataset spawrious_o2o_easy --epochs 10 --batch_size 64
echo "[run_spatial] 큐 종료 $(date '+%F %T')"
