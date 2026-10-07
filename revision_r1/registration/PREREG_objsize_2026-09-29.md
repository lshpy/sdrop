# Pre-registration: object-size experiment on ImageNet-9 backgrounds

Frozen 2026-09-29, before any run at the new scales. Author: Seunghyun Lee.

**Status of the hypothesis.** This is a post hoc hypothesis. It came from the pattern across the pre-registered field and screening results, and it is tested here prospectively. The paper will say so.

**Hypothesis.** Suppressing the dominant channels helps only when the shortcut occupies most of the image, so that it carries most of the layer-4 activation energy. When the object fills the frame, the dominant channels are object channels, and suppressing them does not help or even hurts.

The evidence that suggested it:
- Waterbirds, a small bird on a large scene: works.
- CounterAnimal, animals filling iNaturalist photos: SDrop hurts, E−R −0.85, 0/6.
- ISIC, a small artifact: the gain equals a prior shift.
- in9bg at original scale, objects about 35–45% of the image: null.

**Manipulation.**
- The in9bg benchmark is rebuilt exactly as registered in `PREREG_SCREEN_DATASETS.md`: same splits, same object/background pairs, same correlation ρ = 0.9.
- The only change is that the masked object is cropped to its bounding box, rescaled by a factor s, and pasted at a seeded random position (`prep_in9bg.py --scale s`).
- Scales: s = 1.0 (the existing `checkpoints_screenfull` in9bg runs), 0.5 and 0.3.
- Measured object area (fraction of image), from a 5-per-class sample: about 0.09–0.11 at s = 0.5 and about 0.03 at s = 0.3.

**Design.** Identical to the field and screening runs:
- ImageNet ResNet-50, 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, best-validation checkpoint, `--dump_feats`;
- arms ERM, Energy, Random, Shuffle, Energy top-k, Shuffle top-k at L4, p = 0.1;
- seeds 0–5.
- 72 new runs (`checkpoints_in9scale`). No tuning; no arm or seed is dropped.

**Endpoints and tests.**
- **Primary:** worst-group accuracy over the 81 test cells, as for in9bg.
  - **Primary test:** Energy − Random at s = 0.3, seed-paired two-sided t-test, n = 6. Success = mean > 0 and p < 0.05.
  - This test joins the confirmatory family.
- **Secondary** (exploratory family):
  1. The same test at s = 0.5.
  2. A trend test of Energy − Random across s ∈ {1.0, 0.5, 0.3}: regression of the seed-paired difference on log(object area), slope < 0.
  3. Energy top-k − Shuffle top-k at each scale.
  4. Energy vs ERM + logit adjustment at matched average accuracy, at each scale (two-sided τ).
  5. The L4 background/class ratios of the ERM seed-0 model at each scale (`analyze_units.py`), reported next to the outcome.
- **Reporting:** all outcomes are reported whatever they are. If the primary test fails, the paper states that object size does not explain where SDrop works.
