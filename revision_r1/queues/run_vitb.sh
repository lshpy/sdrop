#!/usr/bin/env bash
# ViT-B/16 (ImageNet 사전학습) on Waterbirds — 마지막 블록 토큰 임베딩(768=채널)에 SDrop. ResNet-50 과 동일 설정.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --self_gamma --layers L4"
for s in 0 1 2; do
  run "waterbirds_vitb_none_rate0.1_L4_vit_b_16_ep30_seed${s}"                       $C --seed $s --method vitb_none
  for m in vitb_sdrop_energy vitb_sdrop_gap vitb_sdrop_random vitb_sdrop_shuffle; do
    run "waterbirds_${m}_rate0.1_L4_vit_b_16_ep30_nmmean_sg_seed${s}"               $C --seed $s --method $m --drop_rate 0.1
  done
done
echo "[run_vitb] 큐 종료 $(date '+%F %T')"
