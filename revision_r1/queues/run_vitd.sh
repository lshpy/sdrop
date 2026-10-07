#!/usr/bin/env bash
# ViT-B/16 MLP-neuron dose sweep (2026-09-24): does the budget-dependence prediction hold on a transformer?
# GAP at 0.2/0.3, energy at 0.3, random control at 0.3, three seeds, dumps for the logit-adjustment control.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --self_gamma --layers L4 --dump_feats --save_dir $CKPT"
for s in 0 1 2; do
  for r in 0.2 0.3; do
    run "waterbirds_vitbm_sdrop_gap_rate${r}_L4_vit_b_16_ep30_nmmean_sg_seed${s}"     $C --seed $s --method vitbm_sdrop_gap    --drop_rate $r
  done
  run "waterbirds_vitbm_sdrop_energy_rate0.3_L4_vit_b_16_ep30_nmmean_sg_seed${s}"     $C --seed $s --method vitbm_sdrop_energy --drop_rate 0.3
  run "waterbirds_vitbm_sdrop_random_rate0.3_L4_vit_b_16_ep30_nmmean_sg_seed${s}"     $C --seed $s --method vitbm_sdrop_random --drop_rate 0.3
done
echo "[run_vitd] 큐 종료 $(date '+%F %T')"
