#!/usr/bin/env bash
# Waterbirds 보강: 베이스라인 5종 시드 3-5 (→6시드) + Guided Dropout const 예산(원논문 스케줄 없이) 6시드
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --num_workers 12 --drop_rate 0.1 --layers L4"
for s in 3 4 5; do
  run "waterbirds_guided_topk_rate0.1_L4_ep30_schgdtopk_seed${s}" $C --seed $s --method guided_topk --schedule gdtopk
  run "waterbirds_guided_dr_rate0.1_L4_ep30_schgdtopk_seed${s}"   $C --seed $s --method guided_dr   --schedule gdtopk
  run "waterbirds_wcd_rate0.1_L4_ep30_seed${s}"                   $C --seed $s --method wcd
  run "waterbirds_dropblock_rate0.1_L4_ep30_seed${s}"             $C --seed $s --method dropblock
  run "waterbirds_spatialdropout_rate0.1_L4_ep30_seed${s}"        $C --seed $s --method spatialdropout
  run "waterbirds_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"   $C --seed $s --method sdrop_random --norm mean
  run "waterbirds_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seed${s}"  $C --seed $s --method sdrop_shuffle --norm mean
  run "waterbirds_sdrop_rate0.1_L4_ep30_nmmean_seed${s}"          $C --seed $s --method sdrop --norm mean
done
for s in 0 1 2 3 4 5; do
  run "waterbirds_guided_topk_rate0.1_L4_ep30_seed${s}" $C --seed $s --method guided_topk --schedule const
  run "waterbirds_guided_dr_rate0.1_L4_ep30_seed${s}"   $C --seed $s --method guided_dr   --schedule const
done
echo "[run_wb_more] 큐 종료 $(date '+%F %T')"
