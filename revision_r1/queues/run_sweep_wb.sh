#!/usr/bin/env bash
# Evidence sweep (2026-09-15): Waterbirds train-correlation sweep, size-matched control,
# class-balanced (SUBY) control with dumps, CelebA SUBY control.
source "$(dirname "$0")/_queue_lib.sh"
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --save_dir $CKPT"
for corr in 0.9 0.75 0.5; do for s in 0 1 2 3 4 5; do
  run "waterbirds_none_rate0.1_none_ep30_corr${corr}_seed${s}"                 $C --seed $s --method none --train_corr $corr
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_corr${corr}_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4 --train_corr $corr
  run "waterbirds_sdrop_random_rate0.1_L4_ep30_nmmean_corr${corr}_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4 --train_corr $corr
done; done
for s in 0 1 2 3 4 5; do   # size-matched control: 10% random subset at the native 95% correlation
  run "waterbirds_none_rate0.1_none_ep30_sub0.1_seed${s}"                      $C --seed $s --method none --subset_frac 0.1
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_sub0.1_nmmean_seed${s}"         $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4 --subset_frac 0.1
  run "waterbirds_sdrop_random_rate0.1_L4_ep30_sub0.1_nmmean_seed${s}"         $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4 --subset_frac 0.1
done
for s in 0 1 2 3 4 5; do   # class-balanced subsampling (label-only), with dumps for LA/DFR
  run "waterbirds_none_rate0.1_none_ep30_cb_seed${s}"                          $C --seed $s --method none --class_balance --dump_feats
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_cb_seed${s}"             $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4 --class_balance --dump_feats
done
CC="--dataset celeba --epochs 8 --lr 0.001 --weight_decay 1e-4 --batch_size 128 --norm mean --num_workers 8 --save_dir $CKPT"
for s in 0 1 2; do
  run "celeba_none_rate0.1_none_ep8_cb_seed${s}"                               $CC --seed $s --method none --class_balance --dump_feats
done
echo "[run_sweep_wb] 큐 종료 $(date '+%F %T')"
