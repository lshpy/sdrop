#!/usr/bin/env bash
# Neurocomputing revision (NEUCOM-D-26-06952, 2026-10-06): the RUN items of CHECKLIST_point_by_point_2026-10-06.md.
# All CIFAR-100 10% data (subset_frac 0.1), ResNet-18, 200 ep, same recipe as the 18-seed headline arm
# (cifar100_sdrop_rate0.1_L4_sub0.1_nmmean_seed*). Seeds pair with the existing L4 sdrop/random/none runs.
#   R2-2  insertion-layer sweep : L1, L2 x {sdrop, random} x 6 seeds (24); L3 top-up seeds 3-5 (6)
#   R2-4  baselines              : dropblock, targeted at L4 x 6 seeds (12)
#   R2-6  gamma/delta            : sdrop L4, one-at-a-time gamma {0.5,2} delta {0.5,2} x 3 seeds (12)
#   R2-6  p_base                 : sdrop/random p=0.5 x 3 seeds (6)   (0.05-0.3 already exist)
# 60 runs. Output checkpoints/ (same folder as the headline runs); weights deleted after each run.
source "$(dirname "$0")/_queue_lib.sh"
r () { run "$@"; rm -f "$CKPT/$1_best.pth"; return 0; }
C="--dataset cifar100 --epochs 200 --subset_frac 0.1 --norm mean --num_workers 6"

# R2-2 layer sweep
for L in L1 L2; do for s in 0 1 2 3 4 5; do
  r "cifar100_sdrop_rate0.1_${L}_sub0.1_nmmean_seed${s}"        $C --seed $s --method sdrop        --drop_rate 0.1 --layers $L
  r "cifar100_sdrop_random_rate0.1_${L}_sub0.1_nmmean_seed${s}" $C --seed $s --method sdrop_random --drop_rate 0.1 --layers $L
done; done
for s in 3 4 5; do
  r "cifar100_sdrop_rate0.1_L3_sub0.1_nmmean_seed${s}"          $C --seed $s --method sdrop        --drop_rate 0.1 --layers L3
  r "cifar100_sdrop_random_rate0.1_L3_sub0.1_nmmean_seed${s}"   $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L3
done

# R2-4 baselines at L4
for s in 0 1 2 3 4 5; do
  r "cifar100_dropblock_rate0.1_L4_sub0.1_seed${s}"             $C --seed $s --method dropblock --drop_rate 0.1 --layers L4
  r "cifar100_targeted_rate0.1_L4_sub0.1_seed${s}"              $C --seed $s --method targeted  --drop_rate 0.1 --layers L4
done

# R2-6 gamma / delta sensitivity (gamma=delta=1 is the headline arm)
for s in 0 1 2; do
  r "cifar100_sdrop_rate0.1_L4_sub0.1_nmmean_g0.5_seed${s}"     $C --seed $s --method sdrop --drop_rate 0.1 --layers L4 --gamma 0.5
  r "cifar100_sdrop_rate0.1_L4_sub0.1_nmmean_g2_seed${s}"       $C --seed $s --method sdrop --drop_rate 0.1 --layers L4 --gamma 2
  r "cifar100_sdrop_rate0.1_L4_sub0.1_nmmean_d0.5_seed${s}"     $C --seed $s --method sdrop --drop_rate 0.1 --layers L4 --delta 0.5
  r "cifar100_sdrop_rate0.1_L4_sub0.1_nmmean_d2_seed${s}"       $C --seed $s --method sdrop --drop_rate 0.1 --layers L4 --delta 2
done

# R2-6 p_base = 0.5
for s in 0 1 2; do
  r "cifar100_sdrop_rate0.5_L4_sub0.1_nmmean_seed${s}"          $C --seed $s --method sdrop        --drop_rate 0.5 --layers L4
  r "cifar100_sdrop_random_rate0.5_L4_sub0.1_nmmean_seed${s}"   $C --seed $s --method sdrop_random --drop_rate 0.5 --layers L4
done
