#!/usr/bin/env bash
# SDrop — 로컬 GPU 머신 세팅 (Linux / macOS / WSL)
#
#   git clone https://github.com/lshpy/sdrop.git && cd sdrop
#   bash setup_local.sh
#
# 끝나면:  bash run_queue.sh
set -e

echo "=============================================="
echo " 1. GPU 확인"
echo "=============================================="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    CUDA_MAJOR=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*' | head -1)
    echo "감지된 CUDA: ${CUDA_MAJOR:-unknown}"
else
    echo "nvidia-smi 없음 — GPU 드라이버를 먼저 설치하세요."
    exit 1
fi

echo
echo "=============================================="
echo " 2. 파이썬 가상환경"
echo "=============================================="
PY=${PYTHON:-python3}
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
echo "venv 준비 완료: $(python --version)"

echo
echo "=============================================="
echo " 3. PyTorch 설치"
echo "=============================================="
# 최신 PyTorch는 sm_70 미만(예: P100, GTX 10xx)을 지원하지 않으므로,
# 컴퓨트 능력을 보고 채널을 고른다.
if [ "${CUDA_MAJOR:-12}" -ge 12 ] 2>/dev/null; then
    CH=cu124
else
    CH=cu118
fi
echo "채널: $CH"
pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$CH"

python - <<'PY'
import torch, sys
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA를 쓸 수 없습니다. 드라이버/설치 채널을 확인하세요.")
cc = torch.cuda.get_device_capability()
print("GPU:", torch.cuda.get_device_name(0), f"| sm_{cc[0]}{cc[1]}")
x = torch.randn(512, 512, device="cuda"); torch.cuda.synchronize()
print("CUDA 연산 확인 OK ->", float((x @ x).sum()))
PY

echo
echo "=============================================="
echo " 4. 나머지 의존성"
echo "=============================================="
pip install -q numpy pandas matplotlib scikit-learn tqdm pyarrow pillow

echo
echo "=============================================="
echo " 5. 동작 확인 (2에폭)"
echo "=============================================="
python train.py --dataset cifar100 --method sdrop_energy --drop_rate 0.1 \
    --layers L4 --epochs 2 --seed 0

echo
echo "=============================================="
echo " 세팅 완료"
echo "=============================================="
echo "다음 실행:"
echo "  source .venv/bin/activate"
echo "  bash run_queue.sh            # 전체 큐"
echo "  bash run_queue.sh longtail   # 롱테일만"
echo "  nohup bash run_queue.sh > queue.log 2>&1 &   # 백그라운드"
