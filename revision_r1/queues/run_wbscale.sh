#!/usr/bin/env bash
# Waterbirds bird-rescaling reverse test (prereg/PREREG_wbscale_2026-09-29.md): 4 scales x 3 arms x seeds 20-31.
source "$(dirname "$0")/_queue_lib.sh"
for s in 20 21 22 23 24 25 26 27 28 29 30 31; do
  for ds in wbscale_s05 wbscale_s10 wbscale_s15 wbscale_s20; do
    C="--dataset $ds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT --seed $s"
    run "${ds}_none_rate0.1_none_ep30_seed${s}"                        $C --method none
    run "${ds}_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"   $C --method sdrop_energy  --drop_rate 0.1 --layers L4 --select topk
    run "${ds}_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"  $C --method sdrop_shuffle --drop_rate 0.1 --layers L4 --select topk
  done
done
echo "[run_wbscale] queue done $(date '+%F %T')"
