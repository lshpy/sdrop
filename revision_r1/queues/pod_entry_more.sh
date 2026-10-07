#!/usr/bin/env bash
set -u
MAX_HOURS="${MAX_HOURS:-6}"
cd /workspace/sdrop || exit 1; mkdir -p logs checkpoints_more data/spawrious
exec > >(tee -a logs/pod_entry_more.log) 2>&1
echo "[more] 시작 $(date '+%F %T') pod=${RUNPOD_POD_ID:-?}"
terminate () { echo "[more] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
cd data/spawrious
# 2026-09-22: a corrupt PNG (quota hit during an earlier extraction) broke every easy/medium run; re-extract both archives over the tree.
for ch in o2o_easy o2o_medium; do
  url=$( [ $ch = o2o_easy ] && echo "https://www.dropbox.com/s/kwhiv60ihxe3owy/spawrious224__o2o_easy.tar.gz?dl=1" || echo "https://www.dropbox.com/s/x03gkhdwar5kht4/spawrious224__o2o_medium.tar.gz?dl=1" )
  echo "[more] 다운로드 $ch $(date '+%T')"; curl -sL -o "$ch.tar.gz" "$url" || { echo "[more] 다운로드 실패 $ch"; exit 1; }
  tar -xzf "$ch.tar.gz" 2>/dev/null; rm -f "$ch.tar.gz"; echo "[more] 압축 해제 완료 $ch $(date '+%T')"
done
python3 - <<'PY'
import glob; from PIL import Image
bad=0
for f in glob.glob('spawrious224/**/*.png', recursive=True):
    try: Image.open(f).verify()
    except Exception: bad+=1; print('[more] corrupt', f)
print('[more] corrupt files:', bad)
PY
cd /workspace/sdrop
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=50m CKPT=./checkpoints_more bash run_more.sh > logs/run_more_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do [ "$(date +%s)" -ge "$deadline" ] && { echo "[more] ${MAX_HOURS}h 초과"; pkill -P "$QPID"; kill "$QPID"; break; }; sleep 120; done
echo "[more] 완료 $(ls checkpoints_more/*_history.csv 2>/dev/null | wc -l)/24"
