# Suppressive Dropout (SDrop)

Seunghyun Lee, Korea University

Code for **Suppressive Dropout (SDrop)**, presented as a demo at xAI 2026 (Late-Breaking Work, Demos and Doctoral Consortium). The demo paper is in [`paper/SDrop_XAI2026_demo.pdf`](paper/SDrop_XAI2026_demo.pdf).

## What SDrop is

SDrop is a channel dropout that, during fine-tuning, drops **each sample's most active channels**. For every input it scores the channels of a feature map from their own activations and drops high-scoring channels with higher probability. This is the reverse of attention-style or weighted channel dropout, which keeps strong channels and drops weak ones. SDrop has no learnable parameters and uses no labels or group annotations. At test time it is the identity.

## Scores and controls

All variants live in `sdrop.py` and are selected with `--method` (and `--select`).

| Method | Score per channel `c` |
|---|---|
| `sdrop_energy` | energy `E_c = 1 - (1 + gamma * Sigma_c)^(-delta)`, where `Sigma_c` is the mean squared activation |
| `sdrop_gap` | magnitude: global average pooled activation (GAP) |
| `sdrop` | full EGPG score `E_c * (1 - P_c)`, where `P_c` is the spatial peakedness |
| `--select topk` | drop the `k = round(p * C)` highest-scoring channels deterministically instead of sampling |

Controls use the same drop budget:

| Method | Control |
|---|---|
| `sdrop_random` | random channels (uniform score) |
| `sdrop_shuffle` | same set of drop probabilities, permuted across channels per sample |
| `sdrop_reverse` | same set of probabilities, the highest scores get the lowest probabilities |
| `sdrop_cross` | each sample receives another sample's probability vector (cross-sample) |

`--norm mean` (default) makes the realised drop rate equal the nominal `--drop_rate`. Comparison baselines (`dropout`, `dropblock`, `wcd`, `focuseddrop`, `guided_topk`, `guided_dr`, `senet`, `cbam`) are in `baselines.py`.

## Datasets and backbones

Datasets (`--dataset`): `waterbirds`, `celeba`, `metashift`, NICO++ pairs (`nicopp` or `nicopp:<classA>,<classB>:<ctx0>,<ctx1>`), `spawrious_o2o_{easy,medium,hard}`, Colored-MNIST (`cmnist_0.95`, `cmnist_0.98`, `cmnist_0.995`), `bar`, `cub200`, `cifar100` (and `cifar100_lt`, `cifar10`, `svhn`, `mnist`, `tinyimagenet`). Data goes under `--data_root` (default `./data`); if a dataset is missing, the error message says where to get it and where to put it.

Backbones: ResNet-18 and ResNet-50 (`--arch`, SDrop inserted after the stages given by `--layers`, e.g. `L4`), and ImageNet-pretrained ViT-B/16 (`vitb_*` methods act on last-block tokens, `vitbm_*` on the MLP hidden neurons of the second-to-last block).

## Installation

```bash
git clone https://github.com/lshpy/sdrop.git && cd sdrop
pip install -r requirements.txt
```

## Using the module

```python
import torch
from sdrop import build_sdrop

drop = build_sdrop('sdrop_energy', 0.1, norm='mean')   # or 'sdrop', 'sdrop_gap', 'sdrop_random', ...
x = torch.relu(torch.randn(8, 2048, 7, 7))             # post-ReLU feature map (B, C, H, W)

drop.train(); y = drop(x)                              # drops each sample's most active channels
drop.eval();  assert torch.equal(drop(x), x)           # identity at test time
```

Insert it after a backbone stage, e.g. `model.layer4 = nn.Sequential(model.layer4, drop)`, or use `model.build_model`, which does this for you.

## Training on Waterbirds

Waterbirds is read from the HuggingFace parquet release `grodino/waterbirds` placed in `data/waterbirds/`. Checkpoints are selected on the official validation split; test worst-group and average accuracy are reported at the end.

```bash
COMMON="--dataset waterbirds --epochs 30 --lr 0.001 --weight_decay 1e-4 --batch_size 64 --seed 0"

# ERM
python train.py $COMMON --method none

# SDrop (energy score) after layer4 of ResNet-50
python train.py $COMMON --method sdrop_energy --drop_rate 0.1 --layers L4

# Random-channel control at the same budget
python train.py $COMMON --method sdrop_random --drop_rate 0.1 --layers L4
```

Add `--dump_feats` to save validation/test logits and penultimate features (`*_dump.npz`) for the scripts in `analysis/`. Run `python train.py --help` for all options.

## Repository layout

```
sdrop.py            SDrop module: scores, selection, controls, schedules
sdrop_vit.py        head/key/token SDrop for ViT trained from scratch
vit_pretrained.py   ImageNet ViT-B/16 with SDrop on tokens or MLP neurons
model.py            ResNet backbones with SDrop / baseline insertion
baselines.py        DropBlock, WCD, FocusedDropout, Guided Dropout, SE, CBAM
dataset.py          dataset loaders and splits
train.py            training entry point
evaluate.py         accuracy, ECE and related metrics
evaluate_groups.py  per-group and worst-group accuracy
analysis/           post-hoc analyses (run from the repo root, see each file's docstring)
paper/              xAI 2026 demo paper
```

## Citation

If you use this code, please cite the demo paper:

```bibtex
@inproceedings{lee2026sdrop,
  title     = {{SDrop}: A Demonstration of Explainable Channel-Selective Regularization for Preserving Rare Features},
  author    = {Lee, Seunghyun},
  booktitle = {Late-Breaking Work, Demos and Doctoral Consortium, xAI 2026},
  series    = {CEUR Workshop Proceedings},
  year      = {2026}
}
```

An extended journal version, with Prof. Luca Longo, is in preparation.

## Contact

Seunghyun Lee, sh200411@korea.ac.kr, Korea University
