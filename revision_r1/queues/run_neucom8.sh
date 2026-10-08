#!/usr/bin/env bash
# Activation-mass-matched random control (Part I confound), 2026-10-07: CIFAR-100 10%, L4, p=0.1, 6 seeds, paired with the headline.
# sdrop_massmatch draws the EGPG mask, measures the squared-activation mass it removes, then drops random channels until the same
# mass is removed. Realised mass and channel counts are printed per epoch ("mass_match:" lines in logs/<id>.log).
source "$(dirname "$0")/_queue_lib.sh"
r () { run "$@"; rm -f "$CKPT/$1_best.pth"; return 0; }
C="--dataset cifar100 --epochs 200 --subset_frac 0.1 --norm mean --num_workers 6"
for s in 0 1 2 3 4 5; do r "cifar100_sdrop_massmatch_rate0.1_L4_sub0.1_nmmean_seed${s}" $C --seed $s --method sdrop_massmatch --drop_rate 0.1 --layers L4; done
