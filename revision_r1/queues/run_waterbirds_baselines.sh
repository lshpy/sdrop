#!/usr/bin/env bash
# Waterbirds 선행연구 대조군 (L4, 3시드) + 핵심 팔 시드 보강(3→6).
#   guided_topk/guided_dr (원논문 gdtopk 스케줄), wcd (강한 채널을 *남기는* 방향), dropblock, spatialdropout
#   none / sdrop_energy / sdrop_gap 시드 3-5 추가 → 6시드
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --num_workers 12"
for s in 0 1 2; do
  run "waterbirds_guided_topk_rate0.1_L4_ep30_schgdtopk_seed${s}" $C --seed $s --method guided_topk --drop_rate 0.1 --layers L4 --schedule gdtopk
  run "waterbirds_guided_dr_rate0.1_L4_ep30_schgdtopk_seed${s}"   $C --seed $s --method guided_dr   --drop_rate 0.1 --layers L4 --schedule gdtopk
  run "waterbirds_wcd_rate0.1_L4_ep30_seed${s}"                   $C --seed $s --method wcd         --drop_rate 0.1 --layers L4
  run "waterbirds_dropblock_rate0.1_L4_ep30_seed${s}"             $C --seed $s --method dropblock   --drop_rate 0.1 --layers L4
  run "waterbirds_spatialdropout_rate0.1_L4_ep30_seed${s}"        $C --seed $s --method spatialdropout --drop_rate 0.1 --layers L4
done
for s in 3 4 5; do
  run "waterbirds_none_rate0.1_none_ep30_seed${s}"                $C --seed $s --method none
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"   $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4 --norm mean
  run "waterbirds_sdrop_gap_rate0.1_L4_ep30_nmmean_seed${s}"      $C --seed $s --method sdrop_gap    --drop_rate 0.1 --layers L4 --norm mean
done
echo "[run_waterbirds_baselines] 큐 종료 $(date '+%F %T')"
