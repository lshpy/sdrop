#!/usr/bin/env bash
# Pre-registered field datasets, full runs (prereg/PREREG_field_datasets_2026-09-29.md): 2 datasets x 6 arms x 6 seeds.
# DATASETS and SEEDS from env; SHARD/NSHARD split the queue across pods.
source "$(dirname "$0")/_queue_lib.sh"
for ds in ${DATASETS:-cct20 isic_artifact}; do
  C="--dataset $ds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT"
  for s in ${SEEDS:-0 1 2 3 4 5}; do
    run "${ds}_none_rate0.1_none_ep30_seed${s}"                          $C --seed $s --method none
    run "${ds}_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"             $C --seed $s --method sdrop_energy  --drop_rate 0.1 --layers L4
    run "${ds}_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"             $C --seed $s --method sdrop_random  --drop_rate 0.1 --layers L4
    run "${ds}_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seed${s}"            $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4
    run "${ds}_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"     $C --seed $s --method sdrop_energy  --drop_rate 0.1 --layers L4 --select topk
    run "${ds}_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"    $C --seed $s --method sdrop_shuffle --drop_rate 0.1 --layers L4 --select topk
  done
done
echo "[run_field] queue done $(date '+%F %T')"
