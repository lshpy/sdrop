# Pre-registration: natural photos selected by the condition (COCO cat/dog × indoor/outdoor, small objects)

Frozen 2026-09-29, before any COCO run. Author: Seunghyun Lee.
Construction: `sdrop_pod_code/NATURAL_SMALLOBJ.md` and `prep_coco_smallobj.py`, fixed as of this date.
- Real COCO 2017 photos.
- The cat or dog covers 1–15% of the image.
- place = indoor/outdoor, taken from the panoptic stuff labels with fixed thresholds.
- Train: 2,022 images with P(place = label) = 0.95.
- Validation: 25 per group. Test: 100 per group.

**Why this dataset.**
- It was chosen *by the condition*: a small object on a large background, in natural photos, before any run.
- It is the small-object counterpart of the MetaShift cat/dog task, where SDrop was null.
- Prediction: **gain**.

**Primary estimator, fixed from now on for every new experiment** (reviewer advice against forking paths): Energy top-k vs Shuffle top-k at L4, p = 0.1. Energy (Bernoulli) − Random is secondary.

**Design.**
- Arms: ERM, Energy, Random, Shuffle, Energy top-k, Shuffle top-k. All at L4, p = 0.1, `--norm mean`.
- Seeds: 0–11 (12 seeds).
- Protocol: ResNet-50, 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, best-validation checkpoint (validation worst-group accuracy is not used for selection; the selection rule is the one already in `train.py`), `--dump_feats`.
- No tuning. No arm or seed is dropped.

**Primary test.** Energy top-k − Shuffle top-k on test worst-group accuracy, seed-paired, two-sided t-test, n = 12. Success = mean > 0 and p < 0.05.

**Confirmatory family.** This test joins the confirmatory family. The family is every registered primary to date:
1. Waterbirds headline
2. ISIC
3. CCT20 (p = 1, degenerate)
4. in9bg
5. CounterAnimal
6. FairFace
7. ViT T6
8. object-size ×0.3 primary
9. R1
10. R2
11. this test

BH at q = 0.05 over all 11. The Waterbirds reverse test, if registered later, will be added and the family size stated.

**Secondary tests** (exploratory family):
- Energy − Random.
- Energy top-k − ERM.
- Energy top-k vs ERM + logit adjustment at matched average accuracy, with a two-sided τ sweep.
- The label-based L4 diagnostic of ERM seed 0.
- The XAI background-energy share of ERM seed 0 using the test masks.

**Stated confounds.**
- People co-occur with outdoor dogs.
- COCO has near-duplicate bursts; these will be counted after the build and reported.
- The validation set is small (100 images).

**Reporting.** Every outcome is reported, whatever it is.
