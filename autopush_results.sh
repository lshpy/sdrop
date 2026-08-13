#!/usr/bin/env bash
# 완료된 런의 history CSV를 주기적으로 results-a3000 브랜치에 올린다.
# run_queue.sh와 나란히 돌린다:
#     nohup bash autopush_results.sh > autopush.log 2>&1 &
#
# checkpoints/ 는 .gitignore 대상이라 -f 로 강제 추가한다. .pth 는 개당 45MB라
# 올리지 않는다 — summarize.py 가 읽는 것은 CSV 뿐이다.
set -u
BRANCH=results-a3000
INTERVAL=${INTERVAL:-1800}

while true; do
    n=$(ls checkpoints/*_history.csv 2>/dev/null | wc -l)
    git add -f checkpoints/*_history.csv 2>/dev/null || true

    if git diff --cached --quiet 2>/dev/null; then
        echo "[$(date '+%F %T')] 새 결과 없음 (CSV ${n}개)"
    else
        git commit -q -m "Results from RTX A3000 run: ${n} histories ($(date '+%F %H:%M'))" || true
        if git push -q origin "HEAD:${BRANCH}" 2>/dev/null; then
            echo "[$(date '+%F %T')] 푸시 완료 — CSV ${n}개"
        else
            echo "[$(date '+%F %T')] 푸시 실패 (커밋은 로컬에 남음) — CSV ${n}개"
        fi
    fi

    sleep "$INTERVAL"
done
