#!/usr/bin/env bash
# Seed top-up (2026-09-29, seed policy of rewrite_plan/new_outline.md 3.7): Waterbirds ResNet-50 cells below 6 seeds, + CelebA SUBY.
# Arguments are copied verbatim from the queue that produced seeds 0-2 (named per block), so run ids match and existing runs are skipped.
# Only the missing seeds are listed. CKPT is switched per block to the original folder; `run` reads it at call time.
# Weights are deleted after each fresh run (volume space) except checkpoints_mech2, which the reliance / unit analyses load.
# 74 runs. SHARD/NSHARD split the whole list.
source "$(dirname "$0")/_queue_lib.sh"
r () { local h="$CKPT/$1_history.csv" had=0; [ -f "$h" ] && had=1; run "$@"; [ $had = 0 ] && [ -f "$h" ] && rm -f "$CKPT/$1_best.pth"; return 0; }

# 1. Reliance test, headline arms (tab:reliance rows ERM / Random / Energy): run_mech2_wb.sh, seeds 3-5. Weights KEPT.
CKPT=./checkpoints_mech2; mkdir -p $CKPT
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --log_grad_channels --save_dir $CKPT"
for s in 3 4 5; do
  run "waterbirds_none_rate0.1_none_ep30_seed${s}"                 $C --seed $s --method none
  run "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  run "waterbirds_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
done

# 2. Gradient share (fig:gradshare, GRAD-01/02): run_mech3_wb.sh, seeds 3-5.
CKPT=./checkpoints_mech3; mkdir -p $CKPT
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --log_grad_channels --log_grad_clean 8 --save_dir $CKPT"
for s in 3 4 5; do
  r "waterbirds_none_rate0.1_none_ep30_seed${s}"                 $C --seed $s --method none
  r "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  r "waterbirds_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
done

# 3. Weight-decay sweep (new tab:mech rows): run_rev2a.sh / run_rev2d.sh (identical args), seeds 3-5, into checkpoints_rev2a.
CKPT=./checkpoints_rev2a; mkdir -p $CKPT
C="--dataset waterbirds --epochs 30 --lr 0.001 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
for wd in 1e-3 1e-2 1e-1; do for s in 3 4 5; do
  r "waterbirds_none_rate0.1_none_ep30_wd${wd}_seed${s}"              $C --weight_decay $wd --run_tag wd$wd --seed $s --method none
  r "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_wd${wd}_seed${s}" $C --weight_decay $wd --run_tag wd$wd --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
done; done

# 4. L3 rows (tab:sites L3 selective gain, Appendix D): run_waterbirds.sh (L3 arms) and run_hyp.sh (self_gamma), seeds 3-5, ./checkpoints.
CKPT=./checkpoints; mkdir -p $CKPT
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12"
for s in 3 4 5; do
  for m in sdrop sdrop_gap sdrop_random sdrop_energy sdrop_shuffle; do
    r "waterbirds_${m}_rate0.1_L3_ep30_nmmean_seed${s}"  $C --seed $s --method $m --drop_rate 0.1 --layers L3
  done
done
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --drop_rate 0.1 --layers L3 --self_gamma"
for s in 3 4 5; do
  r "waterbirds_sdrop_energy_rate0.1_L3_ep30_nmmean_sg_seed${s}" $C --seed $s --method sdrop_energy
  r "waterbirds_sdrop_rate0.1_L3_ep30_nmmean_sg_seed${s}"        $C --seed $s --method sdrop
done

# 5. Guided Dropout at constant budget (tab:waterbirds -> Appendix D): run_wb_more.sh, top-k seeds 4-5, DR seeds 3-5, ./checkpoints.
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --num_workers 12 --drop_rate 0.1 --layers L4"
for s in 4 5; do r "waterbirds_guided_topk_rate0.1_L4_ep30_seed${s}" $C --seed $s --method guided_topk --schedule const; done
for s in 3 4 5; do r "waterbirds_guided_dr_rate0.1_L4_ep30_seed${s}"   $C --seed $s --method guided_dr   --schedule const; done

# 6. SUBY, Energy on the class-balanced set (L1035, 1/5 seeds): run_sweep_wb.sh, seed 5.
CKPT=./checkpoints_sweep; mkdir -p $CKPT
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --save_dir $CKPT"
r "waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_cb_seed5"   $C --seed 5 --method sdrop_energy --drop_rate 0.1 --layers L4 --class_balance --dump_feats

# 7. From-scratch ResNet-50 (tab:predict row, "pretrained backbone is part of the condition"): run_scratch_wb.sh, ERM 4-5, Energy/Random 3-5.
CKPT=./checkpoints_scratch; mkdir -p $CKPT
C="--dataset waterbirds --epochs 30 --lr 0.01 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --no_pretrained --dump_feats --save_dir $CKPT"
for s in 4 5; do r "waterbirds_none_rate0.1_none_ep30_scratch_seed${s}" $C --seed $s --method none; done
for s in 3 4 5; do
  r "waterbirds_sdrop_energy_rate0.1_L4_ep30_scratch_nmmean_seed${s}"    $C --seed $s --method sdrop_energy --drop_rate 0.1 --layers L4
  r "waterbirds_sdrop_random_rate0.1_L4_ep30_scratch_nmmean_seed${s}"    $C --seed $s --method sdrop_random --drop_rate 0.1 --layers L4
done

# 8. CelebA SUBY (tab:celeba last row): run_extra_wb.sh, seeds 3-5.
CKPT=./checkpoints_extra; mkdir -p $CKPT
CC="--dataset celeba --epochs 8 --lr 0.001 --weight_decay 1e-4 --batch_size 128 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
for s in 3 4 5; do
  r "celeba_none_rate0.1_none_ep8_cb_seed${s}"                $CC --seed $s --method none --class_balance
done
echo "[run_topup_wb] queue done $(date '+%F %T')"
