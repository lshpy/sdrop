#!/usr/bin/env bash
# Neurocomputing revision, add-on (2026-10-06): balanced full-data CIFAR-100 seed top-up 3-5 for the null table (R2-1),
# none / random / SDrop at L4 p=0.1, same recipe as the existing seeds 0-2 (norm mean). 9 runs, ~25 min each.
source "$(dirname "$0")/_queue_lib.sh"
r () { run "$@"; rm -f "$CKPT/$1_best.pth"; return 0; }
C="--dataset cifar100 --epochs 200 --norm mean --num_workers 6"
for s in 3 4 5; do
  r "cifar100_none_rate0.1_none_seed${s}"                 $C --seed $s --method none
  r "cifar100_sdrop_random_rate0.1_L4_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
  r "cifar100_sdrop_rate0.1_L4_nmmean_seed${s}"           $C --seed $s --method sdrop        --drop_rate 0.1 --layers L4
done
