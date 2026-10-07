# Pre-registration: does the Waterbirds effect survive a larger backbone?

Frozen: 2026-09-30, before any ResNet-101/152 training run on Waterbirds. Author: Seunghyun Lee.
Purpose: robustness of the headline effect to backbone depth (reviewer defence). This is **not** a test of a new mechanism; the "available-but-unused gap" hypothesis was falsified on existing dumps earlier today (`paper_results/analysis/gap_vs_effect.csv`) and gives no reason to expect a larger effect on a larger backbone.

## 1. Fixed facts (ResNet-50 reference, already run)
- Primary estimator of the paper: top-k Energy − Shuffled top-k, worst-group test accuracy at best-validation epoch.
- ResNet-50, seeds 0–5: top-k 70.9 ± 1.7, Shuffled top-k 60.9, Δ = +10.0 (6/6); ERM 60.6.

## 2. Predictions (recorded before running)
- **P1 (author, primary):** the effect persists on ResNet-101: Δ₁₀₁ = top-k − Shuffled top-k > 0, seed-paired.
- **P2 (user's hypothesis, secondary):** a deeper backbone gives a *larger* effect: Δ₁₀₁ > Δ₅₀ and Δ₁₅₂ > Δ₁₀₁.
  - Falsifier for P2: Δ₁₀₁ ≤ Δ₅₀ (point estimate), or the depth trend slope ≤ 0.
- If P1 fails (Δ₁₀₁ ≤ 0 or p ≥ 0.05), the paper states that the effect is specific to ResNet-50 depth and the headline is qualified accordingly.

## 3. Design (fixed)
- Backbones: torchvision `resnet101`, `resnet152`, IMAGENET1K_V1 weights (same weight family as the ResNet-50 runs). `model.py` 2026-09-30: identical bottleneck structure, L4 = 2048 channels, post-ReLU, so the insertion point and score are unchanged.
- Arms per backbone: ERM (`none`), Energy top-k (`sdrop_energy --select topk`), Shuffled top-k (`sdrop_shuffle --select topk`). p = 0.1, L4, `--norm mean`.
- Protocol identical to `run_final_wb.sh`: 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, AMP, best-validation checkpoint, `--dump_feats` on every arm.
- Seeds 0–5 for every arm and backbone: 2 × 3 × 6 = 36 runs. Queue `sdrop_pod_code/run_backbone.sh`, output `sdrop/checkpoints_backbone/`.
- No hyperparameter is tuned on these runs. No arm, seed, or backbone is added or dropped after seeing results. A crashed run is re-run with the same seed.

## 4. Endpoints and decision rule (fixed)
- **Primary test:** ResNet-101, Δ₁₀₁ = top-k − Shuffled top-k, seed-paired two-sided t-test, n = 6. **Success = mean > 0 and p < 0.05.**
- **Secondary tests** (reported, none can rescue a failed primary):
  1. ResNet-152, same test.
  2. Depth trend: Δ per seed for depth ∈ {50, 101, 152}; one-sided test of slope > 0 (P2). Uses the existing ResNet-50 seeds 0–5 from `checkpoints_final/`.
  3. top-k − ERM, per backbone.
  4. ERM worst-group and DFR worst-group per backbone (`scripts/gap_vs_effect.py` on the ERM dumps), to report whether depth changes the gap.
- **Multiple testing:** the primary joins the confirmatory BH family (`scripts/bh_fdr.py`); the secondaries join the exploratory family.
- **Reporting:** whatever the outcome, the result is reported (Section "Robustness to backbone depth" or Limitations).

## 5. Budget
- ResNet-50 Waterbirds ≈ 3 min/run on RTX 5090; 101/152 ≈ 1.7–2.5×. 36 runs ≈ 3–4 GPU-h ≈ $4–5. Balance 2026-09-30: $61.50.

## Addendum (2026-09-30 13:55 KST, before any run): budget cut
- ResNet-152 removed for budget (18 runs instead of 36). No result had been seen.
- P2 reduces to Δ₁₀₁ > Δ₅₀ (Welch t on 6 vs 6 seed-level Δ); the three-point depth trend is dropped.
- Everything else unchanged.
