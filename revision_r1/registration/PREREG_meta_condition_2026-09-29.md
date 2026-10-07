# Pre-registration: meta-analysis across all pre-registered experiments where the background-dominance condition holds

Frozen 2026-09-29, **before any result of the aerialwb or rescaled-Waterbirds (wbscale) runs has been inspected**.
- aerialwb: 39/72 runs had finished on the volume; none had been read.
- wbscale: 12/144 runs had finished; none had been read.

Author: Seunghyun Lee.

**Question.** Across every pre-registered experiment in which the object is small and the background dominates the image, does selective suppression beat a budget-matched shuffled control on average?

**Estimator.** Energy top-k − Shuffle top-k at L4, p = 0.1. This is the primary estimator fixed on 29 September. The effect is the seed-paired difference in test worst-group accuracy, in percentage points (pp).

**Studies included, fixed now (no others will be added or removed).** Each enters as one study with its registered seeds.

| # | Study | Seeds | Status when frozen |
|---|---|---|---|
| S1 | in9bg ×0.5 (object area 9%) | 0–5 | known: +6.0 |
| S2 | in9bg ×0.5 replication | 6–17 | known: +3.25 |
| S3 | in9bg ×0.3 (object area 3%) | 0–5 | known: +2.33 |
| S4 | COCO cat/dog, small objects (natural photos) | 0–11 | known: +2.25 |
| S5 | aerialwb, remote sensing (object area 13%) | 0–11 | **unknown** |
| S6 | Waterbirds rebuilt, bird ×0.5 (area 3%) | 20–31 | **unknown** |
| S7 | Waterbirds rebuilt, bird ×1.0 (area 13%) | 20–31 | **unknown** |

The in9bg ×0.3 replication used Energy − Random, not the fixed estimator, so it is not included as a separate study.

**Primary test.**
- Random-effects meta-analysis (DerSimonian–Laird) of the 7 study means.
- Each study's standard error is SD(Δ)/√n, taken from its paired differences.
- Pooled effect > 0 with a two-sided p < 0.05 counts as success.
- This test joins the confirmatory family.

**Sensitivity analyses (exploratory family).**
- Fixed-effect (inverse-variance) pooling.
- A pooling over only the studies that were unknown when this was frozen (S5–S7). This is the purely prospective subset, and it is the most important check.
- Leave-one-out.
- The same meta-analysis with Energy − Random.

**Contrast (negative control, exploratory).** The same estimator in the registered "background not dominant" conditions: in9bg ×1.0 (seeds 0–5, known), and Waterbirds rebuilt ×1.5 and ×2.0 (seeds 20–31, unknown). Prediction: pooled effect ≈ 0 and smaller than the dominant-condition pooled effect.

**Transparency.** Four of the seven studies were known when this was frozen. The paper will report the prospective subset S5–S7 separately and will say explicitly that S1–S4 informed the decision to register this meta-analysis.
