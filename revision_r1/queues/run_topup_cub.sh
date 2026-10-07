#!/usr/bin/env bash
# Seed top-up (2026-09-29): CUB-200-2011 (boundary table), every arm 3 -> 6 seeds. Args verbatim from run_cub.sh / run_cub_guided.sh,
# folder ./checkpoints (neither original passed --save_dir). 64 runs x ~16-18 min = ~18 GPU-h: the most expensive group.
# Order: core arms first (the ones a boundary-table row and CUB-01 need), seed-major; CORE_ONLY=1 stops after them (27 runs).
# Weights deleted after each fresh run. SHARD/NSHARD.
source "$(dirname "$0")/_queue_lib.sh"
r () { local h="$CKPT/$1_history.csv" had=0; [ -f "$h" ] && had=1; run "$@"; [ $had = 0 ] && [ -f "$h" ] && rm -f "$CKPT/$1_best.pth"; return 0; }
CKPT=./checkpoints; mkdir -p $CKPT
C="--dataset cub200 --epochs 60 --lr 0.01 --batch_size 64 --norm mean --num_workers 12"
for s in 3 4 5; do
  r "cub200_none_rate0.1_none_ep60_seed${s}"  $C --seed $s --method none
  for L in L4 L3; do
    for m in sdrop sdrop_energy sdrop_random sdrop_shuffle; do
      r "cub200_${m}_rate0.1_${L}_ep60_nmmean_seed${s}"  $C --seed $s --method $m --drop_rate 0.1 --layers $L
    done
  done
done
[ "${CORE_ONLY:-0}" = 1 ] && { echo "[run_topup_cub] CORE_ONLY: stop $(date '+%F %T')"; exit 0; }
for s in 3 4 5; do for L in L4 L3; do
  r "cub200_sdrop_gap_rate0.1_${L}_ep60_nmmean_seed${s}"     $C --seed $s --method sdrop_gap     --drop_rate 0.1 --layers $L
  r "cub200_sdrop_reverse_rate0.1_${L}_ep60_nmmean_seed${s}" $C --seed $s --method sdrop_reverse --drop_rate 0.1 --layers $L
done; done
C="--dataset cub200 --epochs 60 --lr 0.01 --batch_size 64 --norm mean --drop_rate 0.1 --num_workers 12"
r "cub200_guided_topk_rate0.1_L4_ep60_schgdtopk_seed0" $C --seed 0 --method guided_topk --layers L4 --schedule gdtopk
for s in 3 4 5; do for L in L4 L3; do
  r "cub200_guided_topk_rate0.1_${L}_ep60_schgdtopk_seed${s}" $C --seed $s --method guided_topk --layers $L --schedule gdtopk
  r "cub200_guided_topk_rate0.1_${L}_ep60_seed${s}"           $C --seed $s --method guided_topk --layers $L --schedule const
  r "cub200_guided_dr_rate0.1_${L}_ep60_schgdtopk_seed${s}"   $C --seed $s --method guided_dr   --layers $L --schedule gdtopk
  r "cub200_guided_dr_rate0.1_${L}_ep60_seed${s}"             $C --seed $s --method guided_dr   --layers $L --schedule const
done; done
echo "[run_topup_cub] queue done $(date '+%F %T')"
