#!/usr/bin/env bash
# Entry for the pre-registered ResNet-101 Waterbirds robustness queue (run_backbone.sh). Waterbirds is already on the volume.
set -u
MAX_HOURS="${MAX_HOURS:-3}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_backbone
exec > >(tee -a logs/pod_entry_backbone.log) 2>&1
echo "[backbone] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[backbone] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow scipy 2>&1 | tail -1
rm -f logs/waterbirds_*resnet101*.log   # 09-30 첫 시도는 sklearn 부재로 즉시 실패 → 로그만 정리
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=40m CKPT=./checkpoints_backbone nohup bash run_backbone.sh > logs/run_backbone_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[backbone] ${MAX_HOURS}h 초과"; pkill -P "$QPID" 2>/dev/null; kill "$QPID"; break; }; sleep 120; done
echo "[backbone] 완료 $(ls checkpoints_backbone/*_history.csv 2>/dev/null | wc -l)/18"
