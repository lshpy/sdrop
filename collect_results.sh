#!/usr/bin/env bash
# 종료 전 반드시 실행 — 결과를 한 파일로 묶습니다.
set -e
cd /workspace/sdrop
python summarize.py 2>/dev/null || true
tar czf /workspace/sdrop_results.tgz \
  checkpoints/*_history.csv logs/ $(ls summary*.csv 2>/dev/null) 2>/dev/null
ls -lh /workspace/sdrop_results.tgz
echo "로컬에서 받기:  runpodctl receive  또는  RunPod 파일 브라우저로 다운로드"
