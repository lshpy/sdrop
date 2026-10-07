#!/usr/bin/env bash
# Object-size replication (prereg/PREREG_objsize_replication_2026-09-29.md): new seeds 6-17 only.
source "$(dirname "$0")/_queue_lib.sh"
for s in 6 7 8 9 10 11 12 13 14 15 16 17; do
  C="--epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 8 --dump_feats --save_dir $CKPT --seed $s"
  run "in9bg_s03_none_rate0.1_none_ep30_seed${s}"                        --dataset in9bg_s03 $C --method none
  run "in9bg_s03_sdrop_energy_rate0.1_L4_ep30_nmmean_seed${s}"           --dataset in9bg_s03 $C --method sdrop_energy  --drop_rate 0.1 --layers L4
  run "in9bg_s03_sdrop_random_rate0.1_L4_ep30_nmmean_seed${s}"           --dataset in9bg_s03 $C --method sdrop_random  --drop_rate 0.1 --layers L4
  run "in9bg_s05_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"   --dataset in9bg_s05 $C --method sdrop_energy  --drop_rate 0.1 --layers L4 --select topk
  run "in9bg_s05_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seltopk_seed${s}"  --dataset in9bg_s05 $C --method sdrop_shuffle --drop_rate 0.1 --layers L4 --select topk
done
echo "[run_in9rep] queue done $(date '+%F %T')"
