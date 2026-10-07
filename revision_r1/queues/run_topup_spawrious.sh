#!/usr/bin/env bash
# Seed top-up (2026-09-29): Spawrious O2O-Easy / O2O-Hard, ERM / Energy / Random, seeds 3-5 (3 -> 6). Args verbatim from run_spawrious.sh,
# folder checkpoints_spw. (run_more.sh tried Easy seeds 3-5 into checkpoints_more on 9/22 and crashed on corrupt PNGs; none finished.)
# Weights deleted after each fresh run, dumps kept. 18 runs, SHARD/NSHARD.
source "$(dirname "$0")/_queue_lib.sh"
r () { local h="$CKPT/$1_history.csv" had=0; [ -f "$h" ] && had=1; run "$@"; [ $had = 0 ] && [ -f "$h" ] && rm -f "$CKPT/$1_best.pth"; return 0; }
CKPT=./checkpoints_spw; mkdir -p $CKPT
for ch in o2o_easy o2o_hard; do
  C="--dataset spawrious_${ch} --epochs 10 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
  for s in 3 4 5; do
    r "spawrious_${ch}_none_rate0.1_none_ep10_seed${s}"                 $C --seed $s --method none
    r "spawrious_${ch}_sdrop_energy_rate0.1_L4_ep10_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
    r "spawrious_${ch}_sdrop_random_rate0.1_L4_ep10_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  done
done
echo "[run_topup_spawrious] queue done $(date '+%F %T')"
