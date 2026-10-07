#!/usr/bin/env bash
# Neurocomputing revision, add-on (2026-10-06): shuffle control at 10% data, 18 seeds paired with the headline arm
# (R3-1 mechanism: same drop-probability distribution re-assigned to random channels). 18 runs.
source "$(dirname "$0")/_queue_lib.sh"
r () { run "$@"; rm -f "$CKPT/$1_best.pth"; return 0; }
C="--dataset cifar100 --epochs 200 --subset_frac 0.1 --norm mean --num_workers 6"
for s in $(seq 0 17); do
  r "cifar100_sdrop_shuffle_rate0.1_L4_sub0.1_nmmean_seed${s}" $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4
done
