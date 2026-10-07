# Pre-registration: two field datasets (camera traps, dermoscopy)

Frozen 2026-09-29, before any SDrop run on these datasets and before the diagnostic result is known. Author: Seunghyun Lee.

This responds to reviewer items 1 and 4 (L. Longo, 2026-09-28): choose a second dataset in advance with the diagnostic, pre-register the bar, and test on genuine field data.

## 1. Datasets (construction fixed in `sdrop_pod_code/PREREG_DATASETS.md`)
- **CCT20.** Caltech Camera Traps, ECCV'18 benchmark (Beery et al.), from LILA (CDLA-Permissive).
  - 12 classes.
  - Confound: camera location.
  - Splits:
    - train: official, 13,496 images;
    - val: cis_val, 2,485;
    - test: cis_test + trans_test, 35,526, restricted to the 117 (class, location) cells with at least 20 images.
- **ISIC-artifact.** ISIC 2018 with the artifact labels of Bissoto et al. (2020), trap split 1.
  - Classes: benign vs malignant.
  - Confound: ruler present.
  - Splits: train 1,563, val 259, test 772. The ruler–label correlation reverses between train (phi +0.41) and test (−0.67).

## 2. Diagnostic and the prediction it makes
- Statistic: `analyze_units.py --dataset <ds>` on the ERM seed-0 checkpoint. It reports the background (confound) ratio and class ratio at ResNet-50 layer1–4, using the same statistic as Table 11.
- **Prediction rule, fixed now:** SDrop at L4 is predicted to help on a dataset **iff** its L4 background ratio > 1.2 **and** its L4 class ratio ≤ 1.0. On Waterbirds these ratios are 2.01 and 0.90; datasets where it did not help have fallen below the first threshold or above the second.
- The diagnostic output, and the prediction it gives for each dataset, will be written into `prereg/PREREG_field_prediction.md` and uploaded (timestamped) **before** any SDrop result is inspected.
- Both datasets are run in full whatever the prediction. The diagnostic is therefore tested in both directions: a gain where it predicts one, and no gain where it does not.

## 3. Design (fixed)
- Arms, all at L4 with p = 0.1 and `--norm mean`:
  - ERM (`none`);
  - Energy (`sdrop_energy`);
  - Random (`sdrop_random`);
  - Shuffle (`sdrop_shuffle`);
  - Energy top-k (`sdrop_energy --select topk`);
  - Shuffle top-k (`sdrop_shuffle --select topk`).
- Protocol: identical to the Waterbirds main runs — ImageNet ResNet-50, 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, best-validation checkpoint, `--dump_feats`.
- Seeds: 0–5 for every arm (6 seeds). No tuning. No arm or seed is added or dropped after results are seen. A crashed run is re-run with the same seed.

## 4. Endpoints and tests (fixed)
- **Primary endpoints:**
  - CCT20: worst-group accuracy over the 117 test cells (`test_worst_group`).
  - ISIC: min(acc benign/ruler, acc malignant/no-ruler) (`min(test_acc_g01, test_acc_g10)`).
- **Primary test per dataset:** Energy − Random on the primary endpoint, seed-paired, two-sided t-test, n = 6. A **gain** means mean > 0 and p < 0.05.
- **Secondary tests**, each reported; none can rescue a failed primary test:
  1. Energy top-k − Shuffle top-k.
  2. Energy − Shuffle.
  3. Energy − ERM.
  4. Energy (and top-k) vs ERM + logit adjustment at matched average accuracy, following the paper's LA protocol.
  5. CCT20 only: accuracy on unseen locations (trans), and mean accuracy over the location-minority cells.
  6. ISIC only: test AUC.
- **Scoring the diagnostic:** it is correct on a dataset if (predicted help and primary gain) or (predicted no help and no primary gain).
- **Multiple testing:** every test joins the paper's Benjamini-Hochberg family (q = 0.05). Raw p and BH q are both reported.
- **Reporting:** all results are reported in the paper whatever their outcome, including a failure of the diagnostic.
