#!/usr/bin/env bash
# 1) Where does the backbone put the shortcut? (analyze_units.py on the ERM checkpoints)  2) ViT MLP dose sweep.
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_vitd analysis
exec > >(tee -a logs/pod_entry_vitd.log) 2>&1
terminate () { echo "[vitd] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[vitd] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
OUT=analysis/unit_selectivity_2026-09-24.csv
for f in checkpoints_mech2/waterbirds_none_rate0.1_none_ep30_seed0_best.pth \
         checkpoints_mech2/waterbirds_none_rate0.1_none_ep30_seed1_best.pth \
         checkpoints_mech2/waterbirds_none_rate0.1_none_ep30_seed2_best.pth \
         checkpoints_vitt/waterbirds_vitb_none_rate0.1_L4_vit_b_16_ep30_sg_seed0_best.pth \
         checkpoints_vitt/waterbirds_vitb_none_rate0.1_L4_vit_b_16_ep30_sg_seed1_best.pth \
         checkpoints_vitt/waterbirds_vitb_none_rate0.1_L4_vit_b_16_ep30_sg_seed2_best.pth; do
  [ -f "$f" ] && { echo "[vitd] units $f"; timeout 20m python analyze_units.py "$f" "$OUT" || echo "[vitd] units 실패 $f"; } || echo "[vitd] 없음 $f"
done
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
CKPT=./checkpoints_vitd RUN_TIMEOUT=25m nohup bash run_vitd.sh > logs/run_vitd_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[vitd] ${MAX_HOURS}h 초과"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[vitd] 진행 $(date '+%T') $(ls checkpoints_vitd/*_history.csv 2>/dev/null | wc -l)/12"; sleep 300
done
echo "[vitd] 완료 $(ls checkpoints_vitd/*_history.csv 2>/dev/null | wc -l)/12"
