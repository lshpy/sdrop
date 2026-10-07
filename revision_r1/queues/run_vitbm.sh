#!/usr/bin/env bash
# ViT-B/16, 마지막 블록 MLP 은닉 뉴런(3072, GELU 뒤)에 SDrop — CNN post-ReLU 채널의 대응물. 나머지 설정은 run_vitb.sh 와 동일.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --self_gamma --layers L4"
for s in 0 1 2; do
  for m in vitbm_sdrop_gap vitbm_sdrop_energy vitbm_sdrop_random vitbm_sdrop_shuffle; do
    run "waterbirds_${m}_rate0.1_L4_vit_b_16_ep30_nmmean_sg_seed${s}" $C --seed $s --method $m --drop_rate 0.1
  done
done
echo "[run_vitbm] 큐 종료 $(date '+%F %T')"
