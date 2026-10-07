#!/usr/bin/env bash
# Screening diagnostic (2026-09-29, PREREG_SCREEN_DATASETS.md): one ERM seed per screening dataset, then
# analyze_units.py on its best checkpoint -> $CKPT/screen_units.csv. Data: pod_entry_screen_diag.sh.
# Protocols: Waterbirds ERM (run_prereg_diag.sh: 30 ep, lr 1e-3, wd 1e-4, bs 64) for the Waterbirds-format sets;
# the paper's CelebA ERM (run_revision_celeba.sh: 8 ep, bs 128) for the two extra CelebA pairs (163k train images).
# SHARD/NSHARD split the queue (_queue_lib.sh); only the shard that trains a run analyzes it, and the CSV
# append is serialized with flock, so shards never write the same row or two headers.
source "$(dirname "$0")/_queue_lib.sh"
W="--epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
C="--epochs 8 --lr 0.001 --weight_decay 1e-4 --batch_size 128 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
i=0
for spec in in9bg:30:W counteranimal:30:W fairface_age:30:W celeba_wavy:8:C celeba_glasses:8:C; do
  IFS=: read -r ds ep p <<< "$spec"; [ "$p" = W ] && args=$W || args=$C
  id="${ds}_none_rate0.1_none_ep${ep}_seed0"
  mine=$(( i % NSHARD )); i=$(( i + 1 ))           # same assignment as run() below (one run per dataset)
  run "$id" --dataset $ds $args --seed 0 --method none
  [ "$mine" != "$SHARD" ] && continue
  if [ -f "$CKPT/${id}_best.pth" ] && ! grep -q "$id" "$CKPT/screen_units.csv" 2>/dev/null; then
    echo "[진단] $id  $(date '+%F %T')"
    rm -f "$CKPT/screen_units.${id}.csv"                # leftover from an interrupted attempt
    python analyze_units.py "$CKPT/${id}_best.pth" "$CKPT/screen_units.${id}.csv" --dataset $ds 2>&1 | tee "logs/${id}_units.log"
    [ -f "$CKPT/screen_units.${id}.csv" ] && flock "$CKPT/screen_units.lock" bash -c \
      "f='$CKPT/screen_units.csv'; [ -s \"\$f\" ] || head -1 '$CKPT/screen_units.${id}.csv' > \"\$f\"; tail -n +2 '$CKPT/screen_units.${id}.csv' >> \"\$f\""
  fi
done
echo "[run_screen_diag] 큐 종료 shard $SHARD/$NSHARD $(date '+%F %T')"
