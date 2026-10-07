#!/usr/bin/env bash
# Peak GPU memory + wall time per epoch for the overhead table (R3-10): CIFAR-100 full data, ResNet-18, 2 epochs per arm,
# nvidia-smi sampled at 1 Hz. Output analysis/peak_mem_2026-10-06.txt. Self-terminates.
set -u
cd /workspace/sdrop || exit 1; mkdir -p logs analysis checkpoints_mem
TAG="neucom6"; exec > >(tee -a logs/pod_entry_${TAG}.log) 2>&1
terminate () { echo "[$TAG] Terminate $(date '+%F %T')"; sync; curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; echo; }
trap terminate EXIT
pip install -q numpy pandas scipy scikit-learn tqdm pyarrow pillow 2>&1 | tail -1
nvidia-smi --query-gpu=name --format=csv,noheader > analysis/peak_mem_2026-10-06.txt
C="--dataset cifar100 --epochs 2 --norm mean --num_workers 6 --save_dir checkpoints_mem --amp"
for arm in "none --method none" "spatialdropout --method dropout --drop_rate 0.1 --layers L4" "random --method sdrop_random --drop_rate 0.1 --layers L4" "sdrop --method sdrop --drop_rate 0.1 --layers L4" "dropblock --method dropblock --drop_rate 0.1 --layers L4" "targeted --method targeted --drop_rate 0.1 --layers L4"; do
  name=${arm%% *}; args=${arm#* }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > logs/mem_${name}.log 2>/dev/null & SMI=$!
  t0=$(date +%s.%N); python train.py $C $args --seed 0 > logs/mem_${name}_train.log 2>&1; t1=$(date +%s.%N)
  kill $SMI 2>/dev/null; sleep 1
  peak=$(sort -n logs/mem_${name}.log | tail -1); ep=$(grep -o '([0-9.]*s)' logs/mem_${name}_train.log | tail -1)
  echo "$name peak_mem_MiB=$peak wall_s=$(python3 -c "print(round($t1-$t0,1))") last_epoch_time=$ep" | tee -a analysis/peak_mem_2026-10-06.txt
done
echo "[$TAG] done $(date '+%F %T')"
