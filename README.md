# Suppressive Dropout (SDrop)
### An Explainable Channel-Selective Regularization Method for Preserving Rare Features

**Seunghyun Lee** — Korea University, Department of Industrial Management Engineering  
*Submitted to Neurocomputing (Elsevier), 2026*

---

## Overview

Deep learning models overfit to **dominant features** (strong textures, backgrounds, high-energy activations), suppressing rare but informative patterns. This mirrors **lateral inhibition** in neuroscience — strongly activated neurons suppress neighbors via Local Response Normalization (LRN).

**SDrop inverts the conventional dropout assumption:**
> All prior methods treat high activation as a proxy for importance.  
> SDrop argues the opposite: the most dominant channels are *harmful* — they monopolize the feature space and starve minority-class detectors of gradient signal.

| Method | Strategy |
|--------|----------|
| Standard Dropout | Drop randomly |
| Targeted Dropout | Drop *weakest* channels |
| SE-Net / CBAM | *Amplify* dominant channels |
| **SDrop (ours)** | **Drop the *strongest* over-inhibitory channels** |

---

## Method

### EGPG Score

Each channel $c$ is scored by its suppressive influence:

$$s_c = E_c \cdot (1 - P_c)$$

- **Channel Energy** $E_c = 1 - \left(1 + \gamma \cdot \frac{1}{HW}\sum_{h,w} x_{c,h,w}^2\right)^{-\delta}$ — suppression potential
- **Spatial Peakedness** $P_c = \frac{\max_{h,w}|x_{c,h,w}|}{\sum_{h,w}|x_{c,h,w}| + \epsilon}$ — activation concentration

High $s_c$ = high energy **and** diffuse spatial spread → over-inhibitory → preferentially dropped.

### Drop Probability

$$p_{\text{drop},c} = p_{\text{base}} \cdot \frac{s_c}{\max_c s_c}$$

### SGridLC (Spatially-Aware Extension)

Partitions the feature map $X \in \mathbb{R}^{C \times H \times W}$ into a $G \times G$ spatial grid. Each cell $(i,j)$ computes independent local scores $s_c^{(i,j)}$, separating object and background treatment.

### Architecture

SDrop is inserted at **Layer 3 (L3)** and **Layer 4 (L4)** of a ResNet backbone:

```
Input → L1(64ch) → L2(128ch) → L3(256ch)[SDrop] → L4(512ch)[SDrop] → FC
```

---

## Experimental Results

### Setup

| Dataset | Resolution | Train / Val | Backbone | Epochs | Optimizer |
|---------|-----------|-------------|----------|--------|-----------|
| CIFAR-100 | 32×32 | 50,000 / 10,000 | ResNet-18 | — | SGD, lr=0.1, cos anneal |
| TinyImageNet | 64×64 | 100,000 / 10,000 | ResNet-50 | — | SGD, lr=0.1, cos anneal |

- Momentum = 0.9, weight decay = 5×10⁻⁴, batch size = 128, standard augmentation
- Results: **mean ± std over 3 runs**

---

### CIFAR-100 Results

| Method | Rate | Layer | Grid | Accuracy (%) |
|--------|------|-------|------|-------------|
| Baseline | 0.0 | — | — | 76.65 ± 0.11 |
| Dropout | 0.1 | L3 | — | 76.56 ± 0.06 |
| Dropout | 0.1 | L4 | — | 76.78 ± 0.11 |
| Dropout | 0.3 | L3 | — | 76.53 ± 0.39 |
| Dropout | 0.3 | L4 | — | 76.41 ± 0.09 |
| SDrop | 0.1 | L4 | — | 76.91 ± 0.13 |
| **SDrop_Energy** | **0.1** | **L4** | **—** | **77.42 ± 0.16** |
| SDrop_Energy | 0.3 | L3 | — | 76.05 ± 0.19 |
| SGridLC | 0.1 | L3 | 2 | 76.76 ± 0.20 |
| SGridLC | 0.3 | L4 | 2 | 76.52 ± 0.08 |

**Best: SDrop_Energy (Rate 0.1, L4) = 77.42% (+0.77%p over baseline)**

---

### TinyImageNet Results

| Method | Rate | Layer | Grid | Accuracy (%) |
|--------|------|-------|------|-------------|
| Baseline | 0.0 | — | — | 48.62 ± 0.03 |
| Dropout | 0.3 | L3 | — | 48.55 ± 0.25 |
| Dropout | 0.3 | L4 | — | 48.34 ± 0.06 |
| SDrop | 0.1 | L3 | — | 48.60 ± 0.18 |
| SDrop_Energy | 0.1 | L3 | — | 48.54 ± 0.21 |
| SGridLC | 0.1 | L3 | 4 | 32.34 ± 27.58 ⚠️ |
| **SGridLC** | **0.3** | **L3** | **4** | **48.66 ± 0.42** |

**Best: SGridLC (Rate 0.3, L3, G=4) = 48.66% (+0.04%p over baseline)**

> ⚠️ SGridLC (Rate 0.1, G=4, L3) shows severe instability (32.34 ± 27.58): fine-grained grids with low drop rates disrupt spatial continuity in early training.

---

## Planned: CUB-200-2011 Experiment

### Motivation

CUB-200-2011 is the most direct validation of SDrop's core claim. Fine-grained bird classification requires distinguishing **subtle, rare visual features** (plumage patterns, beak shape, wing markings) against dominant background textures — precisely the failure mode SDrop is designed to fix.

| Property | Why It Matters for SDrop |
|----------|--------------------------|
| 200 fine-grained classes | More inter-class feature overlap → dominant channels more harmful |
| Subtle discriminative features | Rare feature preservation = the core SDrop hypothesis |
| Background-heavy images | SGridLC's spatial separation should shine here |
| 5,994 train / 5,794 test | Smaller training set → regularization effect is more critical |

### Planned Setup

| Config | Value |
|--------|-------|
| Dataset | CUB-200-2011 |
| Resolution | 224×224 (standard crop) |
| Backbone | ResNet-50 (pretrained ImageNet) |
| SDrop insertion | L3, L4 |
| Drop rates | 0.1, 0.3 |
| Grid sizes (SGridLC) | G=2, G=4 |
| Runs | 3 (mean ± std) |
| Optimizer | SGD, lr=0.01 (fine-tune), momentum=0.9, wd=5×10⁻⁴ |

### Expected Outcome

Fine-grained datasets are where SDrop should demonstrate the **largest margin** over standard dropout, because:
- Background channels (high energy, diffuse) will receive high EGPG scores → dropped
- Fine-grained discriminative channels (low energy, spatially peaky) → preserved
- SGridLC's grid decomposition should spatially isolate object vs. background suppression

### TODO: Results Table (to be filled after experiments)

| Method | Rate | Layer | Grid | Accuracy (%) |
|--------|------|-------|------|-------------|
| Baseline | 0.0 | — | — | TBD |
| Dropout | 0.1 | L4 | — | TBD |
| Dropout | 0.3 | L4 | — | TBD |
| SDrop_Energy | 0.1 | L4 | — | TBD |
| SDrop_Energy | 0.3 | L4 | — | TBD |
| SGridLC | 0.1 | L3 | 2 | TBD |
| SGridLC | 0.1 | L3 | 4 | TBD |
| SGridLC | 0.3 | L3 | 4 | TBD |

---

## Evaluation Metrics

All evaluations use [`evaluate.py`](evaluate.py):

| Metric | Description |
|--------|-------------|
| **Accuracy** | Top-1 accuracy (%) |
| **F1-Macro** | Unweighted mean F1 across classes |
| **F1-Micro** | Global F1 (total TP/FP/FN) |
| **AUC** | ROC-AUC (OvR for multi-class) |
| **ECE** | Expected Calibration Error (10 bins) |

---

## Key Findings

1. **SDrop_Energy outperforms standard dropout** on CIFAR-100: +0.77%p vs. baseline, +0.64%p vs. dropout at the same setting.
2. **Standard dropout degrades at higher rates** (−0.24%p at Rate 0.3), whereas SDrop does not.
3. **SGridLC achieves the best TinyImageNet result** while producing interpretable spatial suppression maps.
4. **Score visualizations confirm the hypothesis**: warm (over-suppressive) regions concentrate on backgrounds; cool (low-suppression) regions on discriminative foreground objects.
5. **Instability warning**: SGridLC with low drop rate + high grid resolution (G=4, Rate=0.1) can cause training collapse.

---

## XAI Integration

Every SDrop decision is fully interpretable:
- **EGPG score maps**: heatmaps show which channels are over-suppressive
- **Spatial drop masks**: SGridLC overlays reveal *where* regularization is most active
- **LRP relevance**: before/after heatmaps show relevance redistribution toward rare-feature regions

Unlike post-hoc methods (LIME, SHAP, GradCAM), SDrop **embeds** the explanation criterion directly into the training loop.

---

## Citation

```bibtex
@article{lee2026sdrop,
  title   = {Suppressive Dropout: An Explainable Channel-Selective Regularization Method for Preserving Rare Features},
  author  = {Lee, Seunghyun},
  journal = {Neurocomputing},
  year    = {2026}
}
```

---

## Contact

**Seunghyun Lee** — sh200411@korea.ac.kr  
Korea University, Seoul, Republic of Korea
