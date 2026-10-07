#!/usr/bin/env bash
# Seed top-up (2026-09-29): ViT-B/16 Waterbirds rows of tab:vitb that have 3 seeds -> 6. Not the pre-registered site runs (run_vitsite.sh).
# Arguments copied from run_vitb.sh (residual stream), run_vitbm.sh (MLP energy) and run_vitd.sh (MLP dose), seeds 3-5.
# Weights (~330 MB each) are deleted after each fresh run; dumps are kept where the original kept them. 27 runs, SHARD/NSHARD.
source "$(dirname "$0")/_queue_lib.sh"
r () { local h="$CKPT/$1_history.csv" had=0; [ -f "$h" ] && had=1; run "$@"; [ $had = 0 ] && [ -f "$h" ] && rm -f "$CKPT/$1_best.pth"; return 0; }

# Residual stream, last block (run_vitb.sh; no --save_dir, so ./checkpoints). Paired with vitb_none seeds 0-5 (already 6).
CKPT=./checkpoints; mkdir -p $CKPT
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --self_gamma --layers L4"
for s in 3 4 5; do
  for m in vitb_sdrop_energy vitb_sdrop_gap vitb_sdrop_random vitb_sdrop_shuffle; do
    r "waterbirds_${m}_rate0.1_L4_vit_b_16_ep30_nmmean_sg_seed${s}"   $C --seed $s --method $m --drop_rate 0.1
  done
  # MLP neurons, energy score at p_base 0.1 (run_vitbm.sh)
  r "waterbirds_vitbm_sdrop_energy_rate0.1_L4_vit_b_16_ep30_nmmean_sg_seed${s}" $C --seed $s --method vitbm_sdrop_energy --drop_rate 0.1
done

# MLP-neuron dose rows (run_vitd.sh), paired with checkpoints_vitt ERM seeds 0-5.
CKPT=./checkpoints_vitd; mkdir -p $CKPT
C="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 12 --self_gamma --layers L4 --dump_feats --save_dir $CKPT"
for s in 3 4 5; do
  for rt in 0.2 0.3; do
    r "waterbirds_vitbm_sdrop_gap_rate${rt}_L4_vit_b_16_ep30_nmmean_sg_seed${s}"     $C --seed $s --method vitbm_sdrop_gap    --drop_rate $rt
  done
  r "waterbirds_vitbm_sdrop_energy_rate0.3_L4_vit_b_16_ep30_nmmean_sg_seed${s}"     $C --seed $s --method vitbm_sdrop_energy --drop_rate 0.3
  r "waterbirds_vitbm_sdrop_random_rate0.3_L4_vit_b_16_ep30_nmmean_sg_seed${s}"     $C --seed $s --method vitbm_sdrop_random --drop_rate 0.3
done
echo "[run_topup_vit] queue done $(date '+%F %T')"
