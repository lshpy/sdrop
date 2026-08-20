#!/usr/bin/env bash
# 대여 GPU용 실험 큐 — Springer ML 원고의 빈칸을 우선순위 순으로 채웁니다.
#
#   bash run_rented.sh must     # A+B+D  필수 (원고가 '완성형'이 되는 최소선)
#   bash run_rented.sh should   # E+F    권장 (표의 빈 행)
#   bash run_rented.sh nice     # G+H    여유되면
#   bash run_rented.sh all      # 전부
#
# GPU 4대에 나눠 돌리기:
#   for i in 0 1 2 3; do
#     SHARD=$i NSHARD=4 CUDA_VISIBLE_DEVICES=$i nohup bash run_rented.sh must \
#       > logs/shard$i.log 2>&1 &
#   done
#
# 이미 끝난 런은 건너뜁니다. 인스턴스가 죽어도 다시 실행하면 이어서 갑니다.
# ⚠ C(pre/post-softmax)는 train.py에 --vit_score 플래그가 없어 코드 패치가 먼저 필요.
set -u
GROUP="${1:-must}"
SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"
CKPT=./checkpoints
mkdir -p "$CKPT" logs
AMP_FLAG=""; [ "${AMP:-1}" = "1" ] && AMP_FLAG="--amp"

IDX=0
run () {  # run <run_id> <train.py 인자...>
  local id="$1"; shift
  local mine=$(( IDX % NSHARD )); IDX=$(( IDX + 1 ))
  [ "$mine" != "$SHARD" ] && return
  if [ -f "$CKPT/${id}_history.csv" ]; then echo "[건너뜀] $id"; return; fi
  echo "[실행] $id  $(date '+%F %T')"
  python train.py "$@" $AMP_FLAG 2>&1 | tee "logs/${id}.log"
  echo "[완료] $id  $(date '+%F %T')"
}
want () { [[ "$GROUP" == all || "$GROUP" == "$1" ]]; }

C="--dataset cifar100 --epochs 200"
VIT="--epochs 200 --lr 1e-3 --strong_aug --mixup_alpha 0.2 --warmup_epochs 10 --label_smoothing 0.1"

if want must; then
  # A. Table 4 아블레이션 빈칸 (원고 811~812줄 TODO)
  for s in 0 1 2; do
    run "cifar100_sdrop_random_rate0.1_L4_seed${s}" \
        $C --method sdrop_random --drop_rate 0.1 --layers L4 --seed "$s"
    run "cifar100_sdrop_peak_rate0.1_L4_seed${s}" \
        $C --method sdrop_peak   --drop_rate 0.1 --layers L4 --seed "$s"
  done
  # 원고 627줄 \todo{run} — unstructured dropout 베이스라인
  for s in 0 1 2; do
    run "cifar100_dropout_std_rate0.1_L4_seed${s}" \
        $C --method dropout_std --drop_rate 0.1 --layers L4 --seed "$s"
  done
  # D. standard dropout seed 0 history 유실분 복구
  run "cifar100_dropout_rate0.1_L4_seed0" \
      $C --method dropout --drop_rate 0.1 --layers L4 --seed 0
  # B. ViT 시드 3→5 확장 (CI가 0을 걸치는 문제)
  for s in 3 4; do
    run "cifar100_vit_rate0.1_none_seed${s}" \
        --dataset cifar100 --method vit $VIT --seed "$s"
    run "cifar100_sdrop_vit_rate0.3_L3+L4_seed${s}" \
        --dataset cifar100 --method sdrop_vit --drop_rate 0.3 --layers L3 L4 $VIT --seed "$s"
  done
  # C. post-softmax 점수 아블레이션 (원고 3.6절 방어).
  #    비교 대상은 이미 있는 200ep seed0 pre-softmax 런이므로 200ep로 맞춘다.
  #    (train.py의 run_id는 epochs를 포함하지 않아 100ep로 돌리면 기존 파일을 덮어씀)
  run "cifar100_sdrop_vit_rate0.3_L3+L4_scorepost_seed0" \
      --dataset cifar100 --method sdrop_vit --drop_rate 0.3 --layers L3 L4 \
      --vit_score post $VIT --seed 0
fi

if want should; then
  # E. 드롭률 스윕 (100에폭 단축 — 논문에 명시할 것)
  for p in 0.05 0.2 0.3 0.5; do
    run "cifar100_sdrop_energy_rate${p}_L4_seed0" \
        --dataset cifar100 --method sdrop_energy --drop_rate "$p" --layers L4 --epochs 100 --seed 0
  done
  # F. SGridLC seed 0 + 층 위치
  run "cifar100_sgridlc_rate0.3_L3_G4_seed0" \
      $C --method sgridlc --drop_rate 0.3 --layers L3 --grid_size 4 --seed 0
  run "cifar100_sdrop_energy_rate0.1_L3_seed0" \
      $C --method sdrop_energy --drop_rate 0.1 --layers L3 --seed 0
  run "cifar100_sdrop_energy_rate0.1_L3+L4_seed0" \
      $C --method sdrop_energy --drop_rate 0.1 --layers L3 L4 --seed 0
fi

if want nice; then
  # G. 정밀화 옵션 아블레이션 (원고가 "natural ablation"으로 예고한 것)
  run "cifar100_sdrop_rate0.1_L4_nmmean_seed0" \
      $C --method sdrop --drop_rate 0.1 --layers L4 --norm mean --seed 0
  run "cifar100_sdrop_rate0.1_L4_pkentropy_seed0" \
      $C --method sdrop --drop_rate 0.1 --layers L4 --peakedness entropy --seed 0
  run "cifar100_sdrop_rate0.1_L4_sg_seed0" \
      $C --method sdrop --drop_rate 0.1 --layers L4 --self_gamma --seed 0
  # H. ViT 층 위치 스윕 (100에폭)
  run "cifar100_sdrop_vit_rate0.3_L3_seed0" \
      --dataset cifar100 --method sdrop_vit --drop_rate 0.3 --layers L3 \
      --epochs 100 --lr 1e-3 --strong_aug --mixup_alpha 0.2 --warmup_epochs 10 --label_smoothing 0.1 --seed 0
  run "cifar100_sdrop_vit_full_rate0.3_none_seed0" \
      --dataset cifar100 --method sdrop_vit_full --drop_rate 0.3 \
      --epochs 100 --lr 1e-3 --strong_aug --mixup_alpha 0.2 --warmup_epochs 10 --label_smoothing 0.1 --seed 0
fi

echo "=== GROUP=$GROUP SHARD=$SHARD 완료 $(date '+%F %T') ==="
