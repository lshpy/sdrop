#!/usr/bin/env bash
# Pre-registered diagnostic (2026-09-29): one ERM seed per field dataset with the Waterbirds ERM protocol
# (run_revision_wb.sh: 30 epochs, SGD lr 1e-3, wd 1e-4, bs 64), then analyze_units.py on its best checkpoint
# -> $CKPT/prereg_units.csv (per-site background ratio / class ratio). Data: pod_entry_prereg_diag.sh.
source "$(dirname "$0")/_queue_lib.sh"
C="--epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --norm mean --num_workers 6 --dump_feats --save_dir $CKPT"
for ds in cct20 isic_artifact; do
  id="${ds}_none_rate0.1_none_ep30_seed0"
  run "$id" --dataset $ds $C --seed 0 --method none
  if [ -f "$CKPT/${id}_best.pth" ] && ! grep -q "$id" "$CKPT/prereg_units.csv" 2>/dev/null; then
    echo "[진단] $id  $(date '+%F %T')"
    python analyze_units.py "$CKPT/${id}_best.pth" "$CKPT/prereg_units.csv" --dataset $ds 2>&1 | tee "logs/${id}_units.log"
  fi
done
echo "[run_prereg_diag] 큐 종료 $(date '+%F %T')"
