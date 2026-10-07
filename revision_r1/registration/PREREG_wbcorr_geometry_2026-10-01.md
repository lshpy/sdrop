# Pre-registration: does the confound–class entanglement track the training correlation within Waterbirds?

Frozen: 2026-10-01 12:45 KST, before any run. Author: Seunghyun Lee.

## Hypothesis (from WHY_ONLY_WATERBIRDS_2026-10-01.md)
The effect requires that ERM's class axis be entangled with the background axis (cos_py ≥ ≈0.4) with the background axis on the dominant
channels. Within Waterbirds the registered effect already falls with the training correlation (Energy − Random: +5.7 at ρ=0.95 → ≈+0.5 at ρ=0.5;
`checkpoints_hyp` histories, no dumps). If entanglement is the condition, cos_py must fall with ρ as well.

## Manipulation
Waterbirds, ERM only, `--train_corr 0.9 / 0.75 / 0.5` (majority groups subsampled; minority kept), seeds 0–2, `--dump_feats`.
Reference ρ = 0.95: existing ERM dumps (checkpoints_mech2, seeds 0–2): cos_py 0.47, cons_p 0.97, mass_p_top 0.53.

## Predictions (recorded before running)
- P1: cos_py is monotone in ρ over {0.5, 0.75, 0.9, 0.95}: cos_py(0.5) < cos_py(0.75) < cos_py(0.9) ≤ cos_py(0.95), and cos_py(0.5) ≤ 0.25 (in9bg level).
  Test: Spearman ρ between training correlation and per-seed cos_py (12 points), one-sided, p < 0.05, plus the point condition cos_py(0.5) ≤ 0.25.
- Falsifier: cos_py(0.5) ≥ 0.40 (entanglement unchanged while the effect vanished) → entanglement is not the condition.
- Secondary (reported): cons_p and mass_p_top vs ρ (expected roughly constant: the composite structure does not change).

## Design (fixed)
ResNet-50, 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, AMP, best-validation checkpoint. 9 runs → `checkpoints_wbcorr/`. Scored with
`OUT=why_wb_geometry_wbcorr.csv python3 scripts/why_wb_geometry.py wbcorr_0.5 wbcorr_0.75 wbcorr_0.9`. Budget ≈ 9 × 3 min ≈ $1.

## Result (2026-10-01 13:20 KST, scored as frozen; pod khuu0wa19ujlan, 9/9; `paper_results/analysis/why_wb_geometry_wbcorr.csv`)
| ρ | cos_py | cons_p | mass_p_top | align_p | ERM wg |
|---|---|---|---|---|---|
| 0.50 | 0.113 | 0.972 | 0.485 | 0.023 | 75.8 |
| 0.75 | 0.292 | 0.969 | 0.528 | 0.301 | 72.3 |
| 0.90 | 0.405 | 0.968 | 0.559 | 0.494 | 65.3 |
| 0.95 (ref) | 0.457 | 0.969 | 0.558 | 0.586 | 58.8 |
- **P1 SUCCESS**: monotone; Spearman(ρ, cos_py) = +0.93 over 12 seed-points, one-sided p < 0.0001; cos_py(0.5) = 0.11 ≤ 0.25. Falsifier (≥ 0.40) not triggered.
- Secondary as predicted: cons_p and mass_p_top constant (composite structure unchanged); align_p falls with ρ.

## Correction (2026-10-01 15:30 KST, after scoring)
The premise sentence in §Hypothesis quoted the within-Waterbirds effect at ρ = 0.5 as "≈ +0.5"; that number was taken from an analysis note
without checking. Recomputed from `checkpoints_hyp` histories (6 seeds, Energy − Random worst-group): ρ = 0.95 +5.7, 0.9 +3.8 (6/6),
0.75 +2.6 (5/6), 0.5 +2.9 (6/6). The effect shrinks with ρ but does not vanish. The registered test (monotonicity of cos_py, point
condition at 0.5, falsifier) is unaffected and its result stands; the interpretation is graded, not binary: low entanglement (cos_py ≈ 0.1–0.2,
as on in9bg and low-ρ Waterbirds) goes with a +2–3 effect, high entanglement (≥ 0.4) with +6–10.
