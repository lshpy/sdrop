#!/usr/bin/env bash
# ViT-B/16 token-level SDrop (2026-09-23): dominant patch tokens at the input of the last block, plus ERM dumps for the LA control.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --self_gamma --layers L4 --dump_feats --save_dir $CKPT"
for s in 0 1 2 3 4 5; do
  run "waterbirds_vitb_none_rate0.1_L4_vit_b_16_ep30_seed${s}"                       $C --seed $s --method vitb_none
  for m in vitbt_sdrop_energy vitbt_sdrop_gap vitbt_sdrop_random; do
    run "waterbirds_${m}_rate0.1_L4_vit_b_16_ep30_nmmean_sg_seed${s}"               $C --seed $s --method $m --drop_rate 0.1
  done
done
echo "[run_vitt] 큐 종료 $(date '+%F %T')"
