# Suppressive Dropout (SDrop)
**An Explainable Channel-Selective Regularization Method for Preserving Rare Features**

Seunghyun Lee — Korea University  
*Submitted to Neurocomputing (Elsevier), 2026*

---

## Quick Summary

SDrop drops the **most dominant channels** during training instead of random or weak ones.  
The key insight: high-energy, diffuse channels monopolize the feature space via lateral inhibition, starving minority-class detectors of gradient signal.

| Method | What it drops | Score |
|--------|--------------|-------|
| Dropout | random | — |
| Targeted Dropout | *weakest* channels | low magnitude |
| **SDrop** | ***strongest*** channels | EGPG = $E_c(1-P_c)$ |
| **SDropEnergy** | ***strongest*** channels | $E_c$ only |
| **SGridLC** | strongest per spatial cell | EGPG per grid cell |

---

## Repository Structure

```
sdrop-neurocomputing/
├── sdrop.py        ← SDrop, SDropEnergy, SGridLC modules + factory
├── model.py        ← ResNet-18/50 with SDrop at L3/L4
├── dataset.py      ← CIFAR-100, TinyImageNet, CUB-200-2011 loaders
├── train.py        ← training script (CLI)
├── evaluate.py     ← evaluation: Acc, F1, AUC, ECE
└── vis_*.png       ← EGPG score visualizations from the paper
```

---

## Environment Setup

```bash
pip install torch torchvision scikit-learn numpy
```

Tested with: Python 3.10, PyTorch 2.x, CUDA 11.8+

---

## Dataset Setup

### 1. CIFAR-100
**Auto-downloads.** No setup needed.

```bash
python train.py --dataset cifar100 --method none   # downloads automatically
```

---

### 2. TinyImageNet

**Step 1:** Download and extract

```bash
cd data/
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

**Step 2:** Reformat validation set (run once)

```python
from dataset import reformat_tinyimagenet_val
reformat_tinyimagenet_val('./data')
```

Expected structure after reformatting:
```
data/tiny-imagenet-200/
├── train/
│   ├── n01443537/
│   │   └── *.JPEG
│   └── ...
└── val/
    ├── n01443537/      ← created by reformat_tinyimagenet_val()
    │   └── *.JPEG
    └── ...
```

---

### 3. CUB-200-2011 (Priority — add for Neurocomputing submission)

**Step 1:** Download

```
URL: https://www.vision.caltech.edu/datasets/cub_200_2011/
File: CUB_200_2011.tgz  (~1.1 GB)
```

**Step 2:** Extract

```bash
tar -xzf CUB_200_2011.tgz -C data/
```

Expected structure:
```
data/CUB_200_2011/
├── images/
│   ├── 001.Black_footed_Albatross/
│   │   └── Black_Footed_Albatross_*.jpg
│   └── ...  (200 classes)
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
└── classes.txt
```

**No further preprocessing needed.**  
Train/test split is read from `train_test_split.txt` (5,994 train / 5,794 test).

---

## Running Experiments

### Replicate paper results

```bash
# --- CIFAR-100: best result (+0.77%p) ---
python train.py --dataset cifar100 --method sdrop_energy --drop_rate 0.1 --layers L4

# --- CIFAR-100: baseline ---
python train.py --dataset cifar100 --method none

# --- CIFAR-100: standard dropout ---
python train.py --dataset cifar100 --method dropout --drop_rate 0.1 --layers L4
python train.py --dataset cifar100 --method dropout --drop_rate 0.3 --layers L4

# --- TinyImageNet: best result (+0.04%p) ---
python train.py --dataset tinyimagenet --method sgridlc --drop_rate 0.3 --layers L3 --grid_size 4
```

### Run 3 seeds (as in paper: mean ± std)

```bash
for seed in 0 1 2; do
  python train.py \
    --dataset cifar100 \
    --method sdrop_energy \
    --drop_rate 0.1 \
    --layers L4 \
    --seed $seed
done
```

### CUB-200-2011 (new — to be run)

```bash
# Baseline
python train.py --dataset cub200 --method none --epochs 60 --lr 0.01

# SDrop_Energy (recommended first run)
python train.py --dataset cub200 --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 60 --lr 0.01

# SGridLC G=2
python train.py --dataset cub200 --method sgridlc --drop_rate 0.3 --layers L3 --grid_size 2 --epochs 60 --lr 0.01

# SGridLC G=4
python train.py --dataset cub200 --method sgridlc --drop_rate 0.3 --layers L3 --grid_size 4 --epochs 60 --lr 0.01
```

> **Note on CUB:** Uses ImageNet-pretrained ResNet-50 by default (strongly recommended for fine-grained tasks). lr=0.01 instead of 0.1.

---

## Full Experiment Grid

### CIFAR-100 (ResNet-18, 200 epochs, lr=0.1)

| Method | Rate | Layer | Grid | Acc (%) | Status |
|--------|------|-------|------|---------|--------|
| Baseline | 0.0 | — | — | 76.65 ± 0.11 | done |
| Dropout | 0.1 | L4 | — | 76.78 ± 0.11 | done |
| Dropout | 0.3 | L4 | — | 76.41 ± 0.09 | done |
| SDrop | 0.1 | L4 | — | 76.91 ± 0.13 | done |
| **SDrop_Energy** | **0.1** | **L4** | **—** | **77.42 ± 0.16** | done ✓ best |
| SGridLC | 0.1 | L3 | 2 | 76.76 ± 0.20 | done |

### TinyImageNet (ResNet-50, 200 epochs, lr=0.1)

| Method | Rate | Layer | Grid | Acc (%) | Status |
|--------|------|-------|------|---------|--------|
| Baseline | 0.0 | — | — | 48.62 ± 0.03 | done |
| Dropout | 0.3 | L3 | — | 48.55 ± 0.25 | done |
| SDrop_Energy | 0.1 | L3 | — | 48.54 ± 0.21 | done |
| **SGridLC** | **0.3** | **L3** | **4** | **48.66 ± 0.42** | done ✓ best |
| SGridLC | 0.1 | L3 | 4 | 32.34 ± 27.58 | done ⚠️ unstable |

### CUB-200-2011 (ResNet-50 pretrained, 60 epochs, lr=0.01) — TODO

| Method | Rate | Layer | Grid | Acc (%) | Status |
|--------|------|-------|------|---------|--------|
| Baseline | 0.0 | — | — | TBD | pending |
| Dropout | 0.1 | L4 | — | TBD | pending |
| Dropout | 0.3 | L4 | — | TBD | pending |
| SDrop | 0.1 | L4 | — | TBD | pending |
| SDrop_Energy | 0.1 | L4 | — | TBD | pending |
| SDrop_Energy | 0.3 | L4 | — | TBD | pending |
| SGridLC | 0.1 | L3 | 2 | TBD | pending |
| SGridLC | 0.3 | L3 | 2 | TBD | pending |
| SGridLC | 0.3 | L3 | 4 | TBD | pending |

---

## Method Reference

### EGPG Score

$$s_c = E_c \cdot (1 - P_c)$$

**Channel Energy** (suppression potential):
$$E_c = 1 - \left(1 + \gamma \cdot \frac{1}{HW}\sum_{h,w} x_{c,h,w}^2\right)^{-\delta}$$

**Spatial Peakedness** (activation concentration):
$$P_c = \frac{\max_{h,w}|x_{c,h,w}|}{\sum_{h,w}|x_{c,h,w}| + \epsilon}$$

**Drop probability**:
$$p_{\text{drop},c} = p_{\text{base}} \cdot \frac{s_c}{\max_c s_c}$$

High $s_c$ = high energy **and** diffuse spread → over-inhibitory → preferentially dropped.

### SGridLC

Partitions feature map into $G \times G$ cells. Each cell computes independent EGPG scores → independent drop mask. Separates object (low suppression) from background (high suppression) regions.

---

## Checkpoints & Logs

Saved to `./checkpoints/`:
- `{run_id}_best.pth` — best validation accuracy checkpoint
- `{run_id}_history.csv` — per-epoch metrics (loss, acc, F1, AUC, ECE)

Run ID format: `{dataset}_{method}_rate{drop_rate}_{layers}_seed{seed}`  
Example: `cifar100_sdrop_energy_rate0.1_L4_seed0_best.pth`

---

## Citation

```bibtex
@article{lee2026sdrop,
  title   = {Suppressive Dropout: An Explainable Channel-Selective
             Regularization Method for Preserving Rare Features},
  author  = {Lee, Seunghyun},
  journal = {Neurocomputing},
  year    = {2026}
}
```

---

## Contact

Seunghyun Lee — sh200411@korea.ac.kr — Korea University
