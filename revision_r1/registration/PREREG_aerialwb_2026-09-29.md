# Pre-registration: remote-sensing Waterbirds analogue (aerialwb: plane/ship × land/water)

Frozen 2026-09-29, before any aerialwb run. Author: Seunghyun Lee.
Construction: `sdrop_pod_code/AERIALWB.md` and `prep_aerialwb.py`, fixed as of this date.
- **Objects:** iSAID instance masks of planes (label 0) and ships (label 1), cut from DOTA-v1.0 images.
- **Backgrounds:** MLRSNet land or water scenes (place 0 = land, place 1 = water).
- **Object area:** U(0.091, 0.169), mean 0.13, the same as the Waterbirds bird.
- **Correlation:** train ρ = 0.95; validation independent; test has 4 cells × 500 images.
- Splits are made by source image.

**Why this dataset.** It was chosen by the condition (a small object on a large background) in a domain different from every dataset so far. Prediction: **gain**.

**Primary estimator** (fixed for all new experiments): Energy top-k − Shuffle top-k at L4, p = 0.1.

**Design.**
- Arms: ERM, Energy, Random, Shuffle, Energy top-k, Shuffle top-k. All at L4, p = 0.1, `--norm mean`.
- Seeds: 0–11.
- Protocol: ResNet-50, 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, best-validation checkpoint, `--dump_feats`.
- 72 runs. No tuning.

**Primary test.** Energy top-k − Shuffle top-k on test worst-group accuracy, seed-paired, two-sided t-test, n = 12. Success = mean > 0 and p < 0.05.

**Confirmatory family.** This test joins the family, bringing it to 13 tests (the 12 listed in `PREREG_wbscale_2026-09-29.md` plus this one), with BH at q = 0.05.

**Secondary tests** (exploratory family):
- Energy − Random.
- Energy top-k − ERM.
- Energy top-k vs ERM + logit adjustment at matched average accuracy, with a two-sided τ sweep.
- The label-based L4 diagnostic of ERM seed 0.
- The XAI background-energy share of ERM seed 0.

**Stated caveats.**
- It is a composite, like Waterbirds.
- Ground resolution differs between sources.
- "Water" is the scene type, not the fraction of water pixels.
- Licences: academic use (iSAID/DOTA); MLRSNet is CC BY 4.0.

**Reporting.** Every outcome is reported, whatever it is.
