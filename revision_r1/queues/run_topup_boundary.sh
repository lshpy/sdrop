#!/usr/bin/env bash
# Seed top-up (2026-09-29): boundary benchmarks of tab:other / tab:predict below 6 seeds -> 6 (Spawrious and CUB have their own queues).
# NICO++ (run_screen.sh / run_screen2.sh), MetaShift (run_metashift.sh), BAR and Colored-MNIST (run_cmnist_bar.sh); args verbatim,
# missing seeds only, original folders. Weights deleted after each fresh run, dumps kept. 50 runs, SHARD/NSHARD.
source "$(dirname "$0")/_queue_lib.sh"
r () { local h="$CKPT/$1_history.csv" had=0; [ -f "$h" ] && had=1; run "$@"; [ $had = 0 ] && [ -f "$h" ] && rm -f "$CKPT/$1_best.pth"; return 0; }

# NICO++ (checkpoints_screen). ERM seeds 3-5 for all three pairs; sheep/kangaroo Energy/Random seeds 2-5.
CKPT=./checkpoints_screen; mkdir -p $CKPT
C="--dataset nicopp --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for s in 3 4 5; do r "nicopp_none_rate0.1_none_ep30_seed${s}" $C --seed $s --method none; done
for pair in "lion,kangaroo" "sheep,kangaroo"; do
  ds="nicopp:${pair}:grass,water"; tag="nicopp_${pair/,/-}_grass-water"
  C="--dataset $ds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
  for s in 3 4 5; do r "${tag}_none_rate0.1_none_ep30_seed${s}" $C --seed $s --method none; done
done
ds="nicopp:sheep,kangaroo:grass,water"; tag="nicopp_sheep-kangaroo_grass-water"
C="--dataset $ds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for s in 2 3 4 5; do
  r "${tag}_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  r "${tag}_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
done

# MetaShift (checkpoints_metashift): Random and Shuffled seed 5.
CKPT=./checkpoints_metashift; mkdir -p $CKPT
C="--dataset metashift --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
r "metashift_sdrop_random_rate0.1_L4_ep30_nmmean_seed5"       $C --seed 5 --method sdrop_random --drop_rate 0.1 --layers L4
r "metashift_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seed5"      $C --seed 5 --method sdrop_shuffle --drop_rate 0.1 --layers L4

# BAR (checkpoints_cb2): all four arms, seed 5.
CKPT=./checkpoints_cb2; mkdir -p $CKPT
B="--dataset bar --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
r "bar_none_rate0.1_none_ep30_seed5"                          $B --seed 5 --method none
r "bar_sdrop_energy_rate0.1_L4_ep30_nmmean_seed5"             $B --seed 5 --method sdrop_energy --drop_rate 0.1 --layers L4
r "bar_sdrop_random_rate0.1_L4_ep30_nmmean_seed5"             $B --seed 5 --method sdrop_random --drop_rate 0.1 --layers L4
r "bar_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seed5"            $B --seed 5 --method sdrop_shuffle --drop_rate 0.1 --layers L4

# Colored-MNIST (checkpoints_cb2): three strengths x three arms, seeds 3-5.
for b in 0.95 0.98 0.995; do
  C="--dataset cmnist_${b} --epochs 20 --lr 0.1 --weight_decay 5e-4 --batch_size 128 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
  for s in 3 4 5; do
    r "cmnist_${b}_none_rate0.1_none_ep20_seed${s}"                 $C --seed $s --method none
    r "cmnist_${b}_sdrop_energy_rate0.1_L4_ep20_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
    r "cmnist_${b}_sdrop_random_rate0.1_L4_ep20_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  done
done
echo "[run_topup_boundary] queue done $(date '+%F %T')"
