#!/usr/bin/env bash
# RunPod 부팅 스크립트 — Web Terminal에 통째로 복붙하세요.
# PyTorch 템플릿(cuda 12.x)에서 시작한다고 가정합니다.
set -e

cd /workspace
if [ ! -d sdrop ]; then
  git clone https://github.com/lshpy/sdrop.git
fi
cd sdrop

pip install -q numpy pandas matplotlib scikit-learn tqdm pyarrow pillow
python - <<'PY'
import torch
print("GPU :", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")
import os; print("CPU :", os.cpu_count(), "cores")
PY

# CIFAR-100 미리 받아두기 (여러 프로세스가 동시에 받다 충돌하는 것 방지)
python -c "
import torchvision
torchvision.datasets.CIFAR100(root='./data', train=True,  download=True)
torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
print('데이터 준비 완료')
"

# 2에폭 스모크 테스트 — 여기서 죽으면 큐 돌리지 말 것
python train.py --dataset cifar100 --method sdrop_energy --drop_rate 0.1 \
  --layers L4 --epochs 2 --seed 0 --amp

echo
echo "==================================================="
echo " 스모크 테스트 통과. 이제 큐를 띄우세요:"
echo
echo "   mkdir -p logs"
echo "   for i in 0 1 2 3; do"
echo "     SHARD=\$i NSHARD=4 nohup bash run_rented.sh all > logs/shard\$i.log 2>&1 &"
echo "   done"
echo "   tail -f logs/shard0.log"
echo
echo " CPU 코어가 16개 미만이면 NSHARD를 2로 낮추세요."
echo "==================================================="
