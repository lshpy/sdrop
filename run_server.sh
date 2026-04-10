#!/bin/bash
# =============================================================
# run_server.sh — DLMATH 서버 (GPU) 실행 스크립트
# 사용법:
#   bash run_server.sh cifar100
#   bash run_server.sh tinyimagenet
#   bash run_server.sh cub200
#   bash run_server.sh all          # 전체 순차 실행
# =============================================================

set -e
DATASET=${1:-cifar100}

# 로그 디렉토리
mkdir -p logs checkpoints

run_dataset() {
    local DS=$1
    echo "=============================="
    echo " Dataset: $DS"
    echo " Seeds:   0 1 2  (3 runs each)"
    echo "=============================="
    python run_experiments.py \
        --dataset $DS \
        --seeds 0 1 2 \
        --data_root ./data \
        --save_dir ./checkpoints \
        --num_workers 8 \
        2>&1 | tee logs/${DS}_$(date +%Y%m%d_%H%M%S).log
}

if [ "$DATASET" = "all" ]; then
    run_dataset cifar100
    run_dataset tinyimagenet
    # CUB는 데이터 다운로드 후 주석 해제
    # run_dataset cub200
else
    run_dataset $DATASET
fi

echo ""
echo "Done. Summary CSV: checkpoints/summary_${DATASET}.csv"
