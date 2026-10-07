# Pre-registration: the "shared confound axis" condition — ImageNet-9 rebuilt as a 2 × 2 Waterbirds

Frozen: 2026-10-01, before any in9wb run. Author: Seunghyun Lee.

## 1. Hypothesis under test (post hoc, from paper_results/analysis/WHY_ONLY_WATERBIRDS_2026-10-01.md)
Strong-channel suppression (label-free, class-agnostic mask) helps when the confound is **one direction shared across classes**
(cons_p ≈ 1, as on Waterbirds 0.97) and the class axis is entangled with it; it is null when the confound is many-valued and
class-specific (in9bg 9-way: cons_p 0.69, Δ ≈ 0; CounterAnimal: 0.02, harm). Object size is NOT the cause (falsified 9/29).

## 2. Manipulation
- Same IN-9 source, same compositing code (`prep_in9bg.py`, 2026-10-01 flags), object scale ×1.0 (the scale that was null 9-way).
- Objects: IN-9 class 0 (dog) vs 1 (bird); labels 0/1.
- Backgrounds: only_bg_t images of IN-9 class 2 (wheeled vehicle: roads/urban) vs class 8 (fish: water); place 0/1.
- Train P(place = paired background) = 0.95 (dog↔vehicle, bird↔fish); validation place independent; test = every test object on one
  background of each pool → 2 × 2 cells × 100.
- Command: `python prep_in9bg.py --tar bg.tar.gz --out data/in9wb --classes 0,1 --bgs 2,8 --rho 0.95`.

## 3. Predictions (recorded before running)
- **P1 (primary):** Energy top-k − Shuffled top-k on worst-group test accuracy > 0, seeds 0–5, paired two-sided t, **success = mean > 0 and p < 0.05**.
  Expected magnitude if the hypothesis is right: ≥ +5 (Waterbirds-like). **Falsifier for the hypothesis: mean ≤ +2.**
- **Manipulation check (reported, not a success criterion):** cons_p of the ERM features (scripts/why_wb_geometry.py) ≥ 0.9.
- Author's prediction: P1 succeeds. If P1 fails, the shared-axis explanation is recorded as falsified (6th), and the paper keeps "condition unknown".

## 4. Design (fixed)
- ImageNet ResNet-50, 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, AMP, best-validation checkpoint, `--dump_feats`.
- Arms: ERM, Energy top-k, Shuffled top-k; L4, p = 0.1, `--norm mean`. Seeds 0–5. 18 runs → `checkpoints_in9wb/`.
- No tuning; nothing added or dropped after seeing results; a crashed run is re-run with the same seed.

## 5. Tests and family
- Primary joins the confirmatory BH family. Secondary: top-k − ERM; ERM worst-group (should be well below ceiling, else the test is uninformative — reported either way).
- Budget ≈ 18 × 2 min ≈ 40 min ≈ $1.5.

## Result (2026-10-01 12:30 KST, scored as frozen; pod subeoa19q6gr9l, 18/18)
- ERM 92.5 ± 2.7, Energy top-k 93.2 ± 2.6, Shuffled top-k 94.0 ± 3.0 (seeds 0–5). **P1: Δ = −0.83, 1/6, p = 0.52 → FAIL; falsifier (≤ +2) triggered.**
- Manipulation check **failed**: ERM cons_p = 0.83 (< 0.9), cos_py = 0.14 (unchanged from 9-way in9bg). The 2-valued background did not create the Waterbirds geometry.
- ERM worst group 92.5 (minority cell dog-on-water 92.7): at ceiling, so per §5 the test is uninformative about the condition. Recorded as: "reducing the number of background values alone does not produce the entangled shared-axis geometry; dog vs bird is solved from ImageNet features without the shortcut."
