#!/usr/bin/env bash
# Guided Dropout(CVPR'19) 베이스라인 — CUB 결정 실험의 선행연구 대조.
#   guided_topk / guided_dr  × {원논문 스케줄 gdtopk, 공정 예산 const}  × L4/L3 × 시드 0-2
#   → 2×2×2×3 = 24런.  run_cub.sh 와 같은 학습 설정(60ep, lr .01, bs 64, p=.1).
# 해석: EGPG > guided_* 이면 "전역 학습 strength 보다 샘플별 공간 구조가 낫다" 의 근거.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset cub200 --epochs 60 --lr 0.01 --batch_size 64 --norm mean --drop_rate 0.1 --num_workers 12"
for s in 0 1 2; do for L in L4 L3; do
  run "cub200_guided_topk_rate0.1_${L}_ep60_schgdtopk_seed${s}" $C --seed $s --method guided_topk --layers $L --schedule gdtopk
  run "cub200_guided_topk_rate0.1_${L}_ep60_seed${s}"           $C --seed $s --method guided_topk --layers $L --schedule const
  run "cub200_guided_dr_rate0.1_${L}_ep60_schgdtopk_seed${s}"   $C --seed $s --method guided_dr   --layers $L --schedule gdtopk
  run "cub200_guided_dr_rate0.1_${L}_ep60_seed${s}"             $C --seed $s --method guided_dr   --layers $L --schedule const
done; done
echo "[run_cub_guided] 큐 종료 $(date '+%F %T')"
