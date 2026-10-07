# Recorded predictions: screening datasets (addendum section B, registered rule)

Written 2026-09-29, from `checkpoints_screen/screen_units.csv` (ERM seed 0, `analyze_units.py --dataset <ds>`, test split). No SDrop run on these datasets had been started when this was written.

Registered rule: SDrop at L4 helps iff the L4 background ratio > 1.2 and the L4 class ratio ≤ 1.0.

| Dataset | L4 background ratio | L4 class ratio | **Prediction** |
|---|---|---|---|
| in9bg (ImageNet-9 backgrounds) | 1.977 | 1.031 | **no gain**: class ratio > 1.0, a narrow miss |
| counteranimal (iNaturalist animals, common vs counter background) | 1.579 | 0.961 | **gain** (Energy − Random > 0, p < 0.05) |
| fairface_age (age, gender confound) | 1.106 | 1.002 | **no gain** |
| celeba_wavy | 0.936 | 0.919 | **no gain** (added 2026-09-29; uploaded as PREREG_screen_prediction_v2.md at 07:06 UTC = 16:06 KST; no SDrop run exists or was ever started on this dataset) |
| celeba_glasses | 0.901 | 0.913 | **no gain** (as above) |

Layers 1–3, for the record:
- in9bg:
  - background: 0.86 / 1.08 / 1.06
  - class: 1.06 / 0.97 / 0.88
- counteranimal:
  - background: 1.18 / 0.89 / 1.05
  - class: 0.97 / 0.95 / 1.06
- fairface_age:
  - background: 0.87 / 0.99 / 0.92
  - class: 0.69 / 0.95 / 0.88

Full runs follow for in9bg, counteranimal and fairface_age (addendum 2): 6 arms × 6 seeds each, with every outcome reported.
