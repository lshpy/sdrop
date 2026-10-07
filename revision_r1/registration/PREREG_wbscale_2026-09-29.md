# Pre-registration: Waterbirds with the bird rescaled (reverse test of the background-dominance condition)

Frozen 2026-09-29, before any rescaled-Waterbirds run. Author: Seunghyun Lee. Design follows the statistics reviewer's recommendation.

**Data.**
- Waterbirds is re-composed with the original group_DRO recipe and the **original bird/background pairs and splits** (release `metadata.csv`). The only change is the bird, scaled by s about its bounding-box centre and shifted just enough to stay in the frame (`prep_waterbirds_scaled.py`).
- At s = 1.0 the rebuilt images match the release up to JPEG noise (mean |diff| 1.1–1.4/255).
- Mean bird area (fraction of the image) by scale:

| s | bird area |
|---|---|
| 0.5 | 0.033 |
| 1.0 | 0.133 |
| 1.5 | 0.282 |
| 2.0 | 0.423 |

- At s = 2.0, 67% of images lose more than 5% of the bird mass to cropping. This confound is stated in advance.

**Prediction** (background-dominance condition): the selective gain shrinks as the bird grows.

**Primary estimator**, fixed for all new experiments: Energy top-k − Shuffle top-k at L4, p = 0.1.

**Design.**
- Scales: s ∈ {0.5, 1.0, 1.5, 2.0}. Each scale is its own rebuilt dataset: wbscale_s05, wbscale_s10, wbscale_s15, wbscale_s20.
- Arms: ERM, Energy top-k, Shuffle top-k.
- **New seeds 20–31** (12 per scale). These are never pooled with the existing Waterbirds seeds.
- Protocol as for the Waterbirds headline runs: ResNet-50, 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, best-validation checkpoint, `--dump_feats`.
- 144 runs.

**Primary test (one test).** A one-sided linear trend of the seed-paired Δ = (Energy top-k − Shuffle top-k) on log(bird area), pooled over the 4 scales (48 differences). Prediction: slope < 0. α = 0.05.

**Pre-stated falsifier.** If the mean Δ at s = 2.0 is ≥ the mean Δ at s = 1.0, the size account is rejected on Waterbirds.

**Confirmatory family.** This test is added to the confirmatory family of `PREREG_coco_smallobj_2026-09-29.md`, bringing it to 12 tests, with BH at q = 0.05. Any further registered primary is added and the final size is stated.

**Secondary tests** (exploratory family):
- Δ at each scale.
- The s = 1.0 vs s = 2.0 pairwise contrast.
- Top-k vs ERM + logit adjustment at matched average accuracy.
- ERM worst-group accuracy per scale, reported to show headroom.
- The sign test.
- The XAI background-energy share of ERM seed 20 at each scale, using `test_masks.npz`.

**Reporting.** Every outcome is reported, whatever it is.
