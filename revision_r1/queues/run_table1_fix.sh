#!/usr/bin/env bash
# Table 1 SE/CBAM 재실행 — BUGS.md A3(lazy init → optimizer 가 파라미터 못 잡음) 로 기존 수치 무효.
# run_fulldata.sh 와 같은 설정(CIFAR-100 full, ResNet-18, 200ep, L4). 드롭 없음(rate0.0).
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset cifar100 --epochs 200 --layers L4 --num_workers 8"
for s in 0 1 2; do
  # 기존 무효 파일과 이름이 겹치지 않도록 _fix 태그 대신 save_dir 분리는 하지 않고,
  # 기존 파일은 pod 에서 먼저 checkpoints/invalid_A3/ 로 옮긴다 (pod_entry_phase2.sh).
  run "cifar100_senet_rate0.0_L4_seed${s}"  $C --method senet --drop_rate 0.0 --seed $s
  run "cifar100_cbam_rate0.0_L4_seed${s}"   $C --method cbam  --drop_rate 0.0 --seed $s
done
echo "[run_table1_fix] 큐 종료 $(date '+%F %T')"
