#!/usr/bin/env bash
# CUB-200-2011 결정 실험 — 공간 항(1−P)이 실제로 변별력을 갖는 해상도에서 EGPG vs GAP.
#
# 배경: CIFAR용 ResNet-18 은 L4 가 4×4(16칸)라 P 가 사실상 상수 → EGPG≈GAP(p=.92).
#       CUB 224px + ResNet-50 은 L4=7×7(49칸), L3=14×14(196칸).
#       새 부위(국소) vs 배경(확산) 이 가설이 예측하는 바로 그 상황.
#
# 팔:  none | sdrop_random | sdrop_gap | sdrop_energy | sdrop(EGPG) | sdrop_shuffle
#      sdrop_shuffle = EGPG 확률 집합을 채널 축에서 permute (점수-채널 연결만 제거)
# 층:  L4(7×7), L3(14×14)     시드: 0 1 2     → 3 + 5×2×3 = 33런
# 순서: 시드-우선. 중간에 끊겨도 완결된 시드 블록이 남도록.
#
#   SHARD=0 NSHARD=1 bash run_cub.sh
source "$(dirname "$0")/_queue_lib.sh"
# ImageNet 사전학습 ResNet-50 파인튜닝 표준: 60ep, lr .01, bs 64  (train.py 헤더 예시와 동일)
C="--dataset cub200 --epochs 60 --lr 0.01 --batch_size 64 --norm mean --num_workers 12"
for s in 0 1 2; do
  run "cub200_none_rate0.1_none_ep60_seed${s}"  $C --seed $s --method none
  for L in L4 L3; do
    for m in sdrop sdrop_gap sdrop_random sdrop_energy sdrop_shuffle; do
      run "cub200_${m}_rate0.1_${L}_ep60_nmmean_seed${s}"  $C --seed $s --method $m --drop_rate 0.1 --layers $L
    done
  done
done
# 여유가 있으면: 억제 방향 대조 (고득점 채널에 최저 확률)
for s in 0 1 2; do for L in L4 L3; do
  run "cub200_sdrop_reverse_rate0.1_${L}_ep60_nmmean_seed${s}" $C --seed $s --method sdrop_reverse --drop_rate 0.1 --layers $L
done; done
echo "[run_cub] 큐 종료 $(date '+%F %T')"
