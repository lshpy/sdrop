#!/usr/bin/env bash
# Reviewer item 3: backbone diagnostic on every ViT-B/16 block and site, all 6 ERM seeds (inference only).
set -u
MAX_HOURS="${MAX_HOURS:-2}"
cd /workspace/sdrop || exit 1; mkdir -p logs analysis
exec > >(tee -a logs/pod_entry_vitall.log) 2>&1
terminate () { echo "[vitall] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[vitall] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
OUT=analysis/vit_allblocks_2026-09-29.csv
CK=""
for s in 0 1 2 3 4 5; do
  f=checkpoints_vitt/waterbirds_vitb_none_rate0.1_L4_vit_b_16_ep30_sg_seed${s}_best.pth
  [ -f "$f" ] && CK="$CK $f" || echo "[vitall] missing $f"
done
timeout $(( MAX_HOURS * 3600 - 300 ))s python analyze_units_vit_all.py "$OUT" $CK || echo "[vitall] analysis failed or timed out"
echo "[vitall] done $(date '+%F %T') rows=$(wc -l < "$OUT" 2>/dev/null)"
