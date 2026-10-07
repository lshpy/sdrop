#!/usr/bin/env bash
# Seed top-up, Waterbirds ResNet-50 + CelebA SUBY (run_topup_wb.sh, 74 runs, ~4.2 GPU-h on a 5090), then the post-hoc analyses
# that depend on them. Sharded by SHARD/NSHARD (env); every shard runs the analyses, which only fire once their inputs are complete.
set -u
MAX_HOURS="${MAX_HOURS:-5}"
cd /workspace/sdrop || exit 1; mkdir -p logs analysis
TAG="topup_wb_s${SHARD:-0}"
exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
echo "[$TAG] start $(date '+%F %T') pod=${RUNPOD_POD_ID:-?} shard=${SHARD:-0}/${NSHARD:-1}"
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow huggingface_hub 2>&1 | tail -1
# Datasets: both are normally on the volume; fetch from HF only if missing (same code as pod_entry_phase2.sh / pod_entry_celeba.sh).
for spec in "waterbirds grodino/waterbirds" "celeba tpremoli/CelebA-attrs"; do set -- $spec
  [ -f data/$1/test-00000-of-00001.parquet ] && continue
  echo "[$TAG] downloading $1"; python - "$1" "$2" <<'PY' || { echo "[$TAG] $1 download failed"; exit 1; }
import sys, shutil, pathlib; from huggingface_hub import snapshot_download
name, repo = sys.argv[1:]; d = snapshot_download(repo, repo_type='dataset', allow_patterns='data/*', local_dir=f'data/_{name}_tmp')
dst = pathlib.Path('data', name); dst.mkdir(exist_ok=True)
for p in pathlib.Path(d, 'data').glob('*.parquet'): shutil.copy(p, dst / p.name)
PY
done
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
RUN_TIMEOUT=20m nohup bash run_topup_wb.sh > logs/run_${TAG}_queue.log 2>&1 & QPID=$!
while kill -0 "$QPID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[$TAG] ${MAX_HOURS}h exceeded"; pkill -P "$QPID" 2>/dev/null; kill "$QPID" 2>/dev/null; break; }
  echo "[$TAG] progress $(date '+%T') $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs finished on this shard"; sleep 300
done
echo "[$TAG] queue finished: $(grep -c '^\[완료\]' logs/run_${TAG}_queue.log) runs on this shard"

# --- post-hoc analyses (each fires only when all six seeds of its inputs exist and its output does not) ---
# Reliance test on the headline arms, seeds 0-5 (tab:reliance rows ERM / Random / Energy). Seeds 0-2 must reproduce reliance_2026-09-24.csv.
M=checkpoints_mech2; REL=analysis/reliance_headline6_topup.csv
CK=$(ls $M/waterbirds_none_rate0.1_none_ep30_seed[0-5]_best.pth $M/waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seed[0-5]_best.pth $M/waterbirds_sdrop_random_rate0.1_L4_ep30_nmmean_seed[0-5]_best.pth 2>/dev/null)
if [ "$(echo $CK | wc -w)" -eq 18 ] && [ ! -f $REL ]; then
  timeout 60m python analyze_reliance.py data/waterbirds/test-00000-of-00001.parquet $REL $CK
else echo "[$TAG] reliance skipped ($(echo $CK | wc -w)/18 checkpoints, or $REL exists)"; fi
# Gradient share (fig:gradshare, GRAD-01/02): rewrites checkpoints_mech3/gradclean_summary.csv over all seeds.
if [ "$(ls checkpoints_mech3/waterbirds_*_gradclean.csv 2>/dev/null | wc -l)" -eq 18 ]; then
  python analyze_gradclean.py checkpoints_mech3 > analysis/gradclean_topup.txt 2>&1; tail -3 analysis/gradclean_topup.txt
else echo "[$TAG] gradclean skipped ($(ls checkpoints_mech3/waterbirds_*_gradclean.csv 2>/dev/null | wc -l)/18)"; fi
# tab:sites ResNet rows (currently ERM seeds 0-2 only): same analysis on the new ERM seeds 3-5 from mech2. One writer only.
if [ "${SHARD:-0}" = 0 ]; then for s in 3 4 5; do
  f=$M/waterbirds_none_rate0.1_none_ep30_seed${s}_best.pth
  [ -f "$f" ] && ! grep -qs "seed${s}" analysis/unit_selectivity_topup.csv && timeout 20m python analyze_units.py "$f" analysis/unit_selectivity_topup.csv
done; fi
# tab:predict "from scratch" row: diagnostic ratio over the six ERM dumps.
[ "$(ls checkpoints_scratch/waterbirds_none_rate0.1_none_ep30_scratch_seed*_dump.npz 2>/dev/null | wc -l)" -eq 6 ] && \
  python analyze_selectivity.py 'checkpoints_scratch/waterbirds_none_rate0.1_none_ep30_scratch_seed*_dump.npz' > analysis/selectivity_scratch_topup.txt 2>&1
echo "[$TAG] done $(date '+%F %T')"
