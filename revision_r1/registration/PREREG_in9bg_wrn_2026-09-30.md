# Pre-registration: confirmation of the in9bg ×1.0 gain on Wide-ResNet-101-2

Frozen: 2026-09-30 17:25 KST, before any run on seeds 3–8. Author: Seunghyun Lee.

## 1. What the screen showed (seeds 0–2, exploratory; PLAN_backbone_screen_2026-09-30.md)
- in9bg ×1.0 (ImageNet-9 composite, object area 37%, 9 classes, place = background class), Wide-ResNet-101-2, L4 channels, p = 0.1:
  ERM 70.7, Energy top-k 74.7, Shuffled top-k 70.7 → top-k − Shuffled top-k = +4.0, 3/3 (+4, +5, +3).
- ResNet-50 on the same data: +0.2 (registered null, PREREG_objsize_2026-09-29.md).
- The screen seeds are NOT used for the confirmation below.

## 2. Prediction (recorded before running)
- **P1 (primary):** on new seeds 3–8, top-k − Shuffled top-k > 0 on in9bg ×1.0 with Wide-ResNet-101-2.
- **P2 (secondary):** the gain exceeds the ResNet-50 gain: Δ_WRN > Δ_R50 (+0.2; R50 seeds 0–5 from checkpoints_in9scale / rev2).
- Falsifier: mean Δ ≤ 0 or p ≥ 0.05 → the screen result is recorded as a non-replicated exploratory finding.

## 3. Design (fixed)
- Arms: ERM, Energy top-k, Shuffled top-k; `--arch wide_resnet101_2`, L4, p = 0.1, `--norm mean`; 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, AMP, best-validation checkpoint, `--dump_feats`.
- Seeds 3, 4, 5, 6, 7, 8 for every arm: 18 runs. Queue `run_in9wrn.sh`, output `checkpoints_in9wrn/`.
- No tuning; no arm or seed added or dropped after seeing results; crashed run re-run with the same seed.

## 4. Endpoints and decision rule (fixed)
- Primary endpoint: worst-group test accuracy at the best-validation epoch.
- Primary test: seed-paired two-sided t-test on top-k − Shuffled top-k, n = 6. **Success = mean > 0 and p < 0.05.**
- Secondary: top-k − ERM (same test); P2 via Welch t on seed-level Δ (WRN seeds 3–8 vs R50 seeds 0–5).
- The primary joins the confirmatory BH family; secondaries join the exploratory family.
- Reported whatever the outcome. If confirmed, the paper adds a "backbone capacity" subsection: the composite large-object condition that is null at ResNet-50 becomes positive at 5× capacity, while real-photo datasets (COCO, CounterAnimal) do not change.

## 5. Budget
- ≈ 3 min/run → ≈ 1 h ≈ $2. Balance ≈ $48.
