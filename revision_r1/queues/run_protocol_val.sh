#!/usr/bin/env bash
# 감사 ⑤ 프로토콜 재실행 — 훈련 집합 안에서 층화 10% validation 으로 checkpoint 선택, test 는 최종 1회.
# 핵심 셀: CIFAR-100 10%, ResNet-18, L4, p=.1, 200ep. none / sdrop / sdrop_random / sdrop_gap × 시드 0-5 = 24런.
# history 의 'acc' 는 validation, 'test_acc' 가 보고 수치.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset cifar100 --epochs 200 --subset_frac 0.1 --val_frac 0.1 --norm mean --layers L4 --num_workers 8"
for s in 0 1 2 3 4 5; do
  run "cifar100_none_rate0.1_none_sub0.1_val0.1_seed${s}"           $C --seed $s --method none
  run "cifar100_sdrop_rate0.1_L4_sub0.1_val0.1_nmmean_seed${s}"      $C --seed $s --method sdrop        --drop_rate 0.1
  run "cifar100_sdrop_random_rate0.1_L4_sub0.1_val0.1_nmmean_seed${s}" $C --seed $s --method sdrop_random --drop_rate 0.1
  run "cifar100_sdrop_gap_rate0.1_L4_sub0.1_val0.1_nmmean_seed${s}"  $C --seed $s --method sdrop_gap    --drop_rate 0.1
done
echo "[run_protocol_val] 큐 종료 $(date '+%F %T')"
