#!/usr/bin/env bash
# Pre-registered ViT-B sites (prereg/PREREG_ViT_sites_2026-09-29.md). SITES and SEEDS come from the env so the
# same script can be split across pods:  SITES="vitbt6" SEEDS="0 1 2" bash run_vitsite.sh
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --self_gamma --layers L4 --dump_feats --save_dir $CKPT"
for s in ${SEEDS:-0 1 2 3 4 5}; do
  for site in ${SITES:-vitbt6 vitbh9}; do
    for k in energy random shuffle; do
      m="${site}_sdrop_${k}"
      run "waterbirds_${m}_rate0.1_L4_vit_b_16_ep30_nmmean_sg_seed${s}" $C --seed $s --method $m --drop_rate 0.1
    done
  done
done
echo "[run_vitsite] queue done $(date '+%F %T')"
