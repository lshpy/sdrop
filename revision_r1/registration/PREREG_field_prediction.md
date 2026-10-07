# Recorded prediction: field datasets (per PREREG_field_datasets_2026-09-29.md, section 2)

Written 2026-09-29 after the diagnostic pod finished and **before any SDrop result on these datasets was inspected**.

- Source: `checkpoints_prereg/prereg_units.csv` from `analyze_units.py --dataset <ds>` on each ERM seed-0 checkpoint (test split).
- Registered rule: SDrop at L4 helps iff the L4 background ratio > 1.2 and the L4 class ratio ≤ 1.0.

| Dataset | L4 background ratio | L4 class ratio | Rule met? | **Prediction** |
|---|---|---|---|---|
| CCT20 (camera location) | 1.032 | 1.061 | no: background ratio ≤ 1.2 and class ratio > 1.0 | **no gain** (Energy − Random not > 0 at p < 0.05) |
| ISIC-artifact (ruler) | 1.273 | 1.237 | no: class ratio > 1.0 | **no gain** |

Other layers, for the record:
- CCT20 L1–L3 background ratios: 0.90 / 0.97 / 1.04; class ratios: 1.03 / 0.98 / 1.11.
- ISIC L1–L3 background ratios: 1.22 / 1.01 / 1.16; class ratios: 1.07 / 1.03 / 1.24.

Interpretation fixed in advance:
- In neither dataset do the dominant L4 channels carry the confound more than the class.
- The diagnostic therefore predicts that suppressing them will not selectively reduce shortcut reliance on either dataset.
- A primary gain on either dataset would count as a **miss** by the diagnostic.
