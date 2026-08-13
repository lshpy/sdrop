# GPU 머신에서 실험 돌리기 — 복붙 가이드

옮길 파일은 없습니다. 이 저장소만 clone하면 코드·스크립트가 전부 들어 있습니다.

---

## Linux / macOS / WSL

```bash
git clone https://github.com/lshpy/sdrop.git
cd sdrop
bash setup_local.sh          # GPU 확인 → venv → PyTorch → 2에폭 테스트
```

문제 없이 끝나면:

```bash
source .venv/bin/activate
nohup bash run_queue.sh > queue.log 2>&1 &
tail -f queue.log
```

## Windows (PowerShell, WSL 없이)

```powershell
git clone https://github.com/lshpy/sdrop.git
cd sdrop
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas matplotlib scikit-learn tqdm pyarrow pillow
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"

# 동작 확인
python train.py --dataset cifar100 --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 2 --seed 0
```

전체 큐는 WSL이나 Git Bash에서 `bash run_queue.sh`로 돌리는 편이 편합니다.
PowerShell만 쓸 경우 `run_queue.sh` 안의 `python train.py ...` 줄을 그대로
복사해 순서대로 실행해도 됩니다.

---

## 큐 구성

| 그룹 | 명령 | 런 수 | 내용 |
|---|---|---|---|
| longtail | `bash run_queue.sh longtail` | 15 | **최우선.** CIFAR-100-LT(100:1) 4방법×3시드 + 평균보존 정규화 3런 |
| ablation | `bash run_queue.sh ablation` | 18 | Random / Peakedness-only / 원소단위 dropout / SGridLC / 베이스라인 시드 보충 |
| grad | `bash run_queue.sh grad` | 7 | gradient-guided suppress vs amplify, 롱테일 조합 포함 |
| vit | `bash run_queue.sh vit` | 4 | ViT 시드 3·4 확장 |
| sweep | `bash run_queue.sh sweep` | 4 | 드롭률 스윕(100에폭 단축) |
| (전체) | `bash run_queue.sh` | 48 | 위 전부 |

이미 끝난 런은 `checkpoints/*_history.csv` 존재 여부로 판단해 **자동으로
건너뜁니다.** 중단했다가 다시 실행하면 이어서 진행됩니다.

## 결과 확인

```bash
python summarize.py            # 방법별 mean ± std, 실효 드롭률 p_eff 요약
```

각 런은 `checkpoints/<run_id>_history.csv`에 에폭별 정확도·F1·AUC·ECE와
**실효 드롭률 `p_eff`**를 기록합니다. `p_eff`는 명목 drop_rate가 아니라 실제로
마스킹된 유닛 비율이며, 논문에서 공정 비교의 근거로 사용합니다.

## 결과를 다시 가져오려면

CSV는 작으므로 저장소에 그대로 올리는 편이 가장 간단합니다.

```bash
git checkout -b results-$(date +%m%d)
git add checkpoints/*_history.csv
git commit -m "Add results from local GPU run"
git push -u origin HEAD
```

---

## 참고

- 전원 절전을 꺼두세요. 며칠 돌아갑니다.
- VRAM이 8GB 미만이면 `--batch_size 64`를 붙이세요(결과에 큰 영향 없음).
- GPU가 sm_70 미만(P100, GTX 10xx 등)이면 최신 PyTorch가 지원하지 않으므로
  `setup_local.sh`가 자동으로 cu118 채널을 씁니다.
- 예상 시간: RTX 4090 기준 롱테일 15런 ≈ 8h, 전체 48런 ≈ 30h.
