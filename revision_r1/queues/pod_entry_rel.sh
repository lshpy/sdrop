#!/usr/bin/env bash
# Reliance probe (2026-09-24): background-swap / background-only / object-only predictions of every surviving Waterbirds ResNet-50.
set -u
cd /workspace/sdrop || exit 1; mkdir -p logs analysis
exec > >(tee -a logs/pod_entry_rel.log) 2>&1
terminate () { echo "[rel] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[rel] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
pip install -q numpy pandas scikit-learn pyarrow pillow 2>&1 | tail -1
CK=$(ls checkpoints_mech2/waterbirds_none_rate0.1_none_ep30_seed[0-9]_best.pth checkpoints_mech2/waterbirds_sdrop_{energy,random,cross}_*_best.pth checkpoints_final/waterbirds_*_best.pth checkpoints_mech/waterbirds_*_best.pth 2>/dev/null)
echo "[rel] $(echo $CK | wc -w) checkpoints"
timeout 90m python analyze_reliance.py data/waterbirds/test-00000-of-00001.parquet analysis/reliance_2026-09-24.csv $CK
echo "[rel] 완료 $(date '+%F %T')"
