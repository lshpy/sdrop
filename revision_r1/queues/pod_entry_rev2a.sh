#!/usr/bin/env bash
# ViT-B token/MLP SDrop queue, resume-safe (run_rev2a.sh skips finished run ids). No smoke test: the code path is already validated.
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_rev2a
exec > >(tee -a logs/pod_entry_rev2a.log) 2>&1
terminate () { echo "[rev2a] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[rev2a] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} done=$(ls checkpoints_rev2a/*_history.csv 2>/dev/null | wc -l)"
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
mkdir -p analysis; [ "$(basename $0)" = "pod_entry_rev2a.sh" ] && timeout 15m python analyze_mass.py checkpoints_mech2/waterbirds_none_rate0.1_none_ep30_seed0_best.pth 8 > analysis/mass_removed_2026-09-24.txt 2>&1
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
CKPT=./checkpoints_rev2a RUN_TIMEOUT=20m nohup bash run_rev2a.sh > logs/run_rev2a_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[rev2a] ${MAX_HOURS}h 초과"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[rev2a] 진행 $(date '+%T') $(ls checkpoints_rev2a/*_history.csv 2>/dev/null | wc -l)"; sleep 300
done
echo "[rev2a] 완료 $(ls checkpoints_rev2a/*_history.csv 2>/dev/null | wc -l)"
