#!/usr/bin/env bash
# H-F: L3 self_gamma 보정 (6런) → H-H 채널 선택성 → H-J 테스트 시간 억제
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --drop_rate 0.1 --layers L3 --self_gamma"
for s in 0 1 2; do
  run "waterbirds_sdrop_energy_rate0.1_L3_ep30_nmmean_sg_seed${s}" $C --seed $s --method sdrop_energy
  run "waterbirds_sdrop_rate0.1_L3_ep30_nmmean_sg_seed${s}"        $C --seed $s --method sdrop
done
echo "[H-H] 시작 $(date '+%F %T')"; python analyze_wb_channels.py 2>&1 | tee logs/analyze_wb_channels.log
echo "[H-J] 시작 $(date '+%F %T')"; python analyze_tts.py 2>&1 | tee logs/analyze_tts.log
echo "[run_hyp] 큐 종료 $(date '+%F %T')"
