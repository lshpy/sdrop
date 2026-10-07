---
title: "SDrop revision: pre-registration (frozen 29 September 2026, before any new result)"
author: "Seunghyun Lee"
geometry: margin=2.2cm
fontsize: 10pt
---
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


ewpage

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


ewpage

# Pre-registration: ViT-B/16 suppression sites predicted by the backbone diagnostic

Frozen: 2026-09-29, before any training run at these sites. Author: Seunghyun Lee.
Responds to reviewer item 3 (L. Longo, 2026-09-28): locate where the diagnostic predicts the effect on ViT-B, or reclassify the ViT-B result as an unexplained transfer.

## 1. Diagnostic that selected the sites (already run, results fixed)
- Statistic: `analyze_units_vit_all.py`. It uses the same background/class selectivity ratios as Table 11 (`analyze_units.py`). Dominant units are the top 10% by mean activation; selectivity is max one-vs-rest |AUC − 0.5|.
- Data: Waterbirds test set.
- Models: the 6 ERM ViT-B/16 checkpoints, seeds 0–5 (`checkpoints_vitt/waterbirds_vitb_none_..._seed{0..5}_best.pth`).
- Sites: 12 blocks × 5 sites (residual, MLP mean, MLP |mean|, patch-token norm, attention-head norm).
- Output: `paper_results/raw_pod/analysis/vit_allblocks_2026-09-29.csv`.
- Selection rule: the condition that holds on Waterbirds ResNet-50 L4 (background ratio > 1 and class ratio ≤ 1), required on all 6 seeds, taking the largest mean background ratio within each site family.
  - **Primary site T6:** patch tokens of block 6's output. Background ratio 2.91 ± 0.60, class ratio 0.77 ± 0.11, condition met on 6/6 seeds.
  - **Secondary site H9:** attention heads of block 9. Background ratio 4.41 ± 0.23, class ratio 0.12 ± 0.06, 6/6 seeds. Caveat: only 12 heads, so "dominant" is a single head.
- For reference, the site used in the current manuscript (MLP neurons of block 10) does **not** meet the condition:
  - mean score: background 1.01, class 1.13;
  - magnitude score: background 0.82, class 1.01.
  - The last-block token site used earlier (tokens of block 10's output) also does not: background 1.00, class 0.66.

## 2. Prediction
If the diagnostic locates where suppression acts, dropping the dominant units at T6 (and at H9) will raise worst-group accuracy more than dropping the same number of randomly chosen units.

## 3. Design (fixed)
- Methods:
  - `vitbt6_sdrop_{energy,random,shuffle}`: tokens entering block 7.
  - `vitbh9_sdrop_{energy,random,shuffle}`: heads inside block 9.
  - Implementation: `vit_pretrained.py`, 2026-09-29.
- Settings:
  - drop rate p = 0.1, `--norm mean --self_gamma`;
  - the same protocol as the existing ViT-B runs (`run_vitt3.sh`): 30 epochs, SGD lr 1e-3, momentum as default, wd 1e-4, batch 64;
  - best-validation checkpoint, `--dump_feats`.
- Seeds: 0–5 for every arm (6 seeds). The ERM arm is the existing 6 `vitb_none` seeds.
- No hyperparameter is tuned on these runs. No arm or seed is added or dropped after seeing results. A run that crashes is re-run with the same seed.

## 4. Endpoints and decision rule (fixed)
- **Primary endpoint:** worst-group test accuracy at the best-validation checkpoint.
- **Primary test (T6):** Energy − Random, seed-paired, two-sided t-test, n = 6. **Success = mean difference > 0 and p < 0.05.**
- **Secondary tests** (each reported, none can rescue a failed primary):
  1. H9 Energy − Random, same test.
  2. T6 and H9 Energy − Shuffle.
  3. Energy − ERM.
  4. Energy vs ERM with logit adjustment at matched average accuracy (the paper's LA protocol).
- **Multiple testing:** all of these tests join the paper's Benjamini-Hochberg family (`scripts/bh_fdr.py`, q = 0.05). They are reported with raw p and BH q.
- **Reporting:** whatever the outcome, the result is reported in the paper. If T6 fails, the paper states that the diagnostic does not predict a working site on ViT-B, and the existing MLP result is described as an unexplained empirical transfer.


ewpage

# Addendum to the 2026-09-29 pre-registrations: confirmatory family, screening, comparison baselines

Frozen 2026-09-29, before any result of the ViT-site, field-dataset or screening runs has been inspected. Author: Seunghyun Lee.
This addendum adds to `PREREG_ViT_sites_2026-09-29.md` and `PREREG_field_datasets_2026-09-29.md` and changes nothing in them.

## A. Confirmatory family (reviewer item 5)
- The paper will separate **confirmatory** from **exploratory** tests.
- The confirmatory family is exactly:
  1. The Waterbirds headline test: Energy − ERM, worst-group, 14 paired seeds (0–9, 16–19).
  2. The ViT-B T6 primary test: Energy − Random.
  3. The primary test of each pre-registered field dataset: CCT20 and ISIC-artifact, Energy − Random.
  4. The primary test of each dataset admitted by the screening in section B: Energy − Random on its registered primary endpoint.
- Benjamini-Hochberg at q = 0.05 is applied within the confirmatory family.
- Every other test in the paper forms the exploratory family. It gets its own BH correction and is described in the text as exploratory throughout, not only in the limitations.

## B. Screening of further datasets (to be run after this addendum)
- Every dataset in `sdrop_pod_code/PREREG_SCREEN_DATASETS.md` is screened: one ERM seed-0 run, then the diagnostic.
- Each dataset's primary endpoint is fixed in that file **before** its screening run.
- The prediction rule is the one already registered: helps iff the L4 background ratio > 1.2 and the L4 class ratio ≤ 1.0.
- The predictions are written to `prereg/PREREG_screen_prediction.md` before any SDrop run on those datasets.
- **Every screened dataset is then run in full**, predicted positive or not, with the same arms and 6 seeds as the field datasets, and **every one is reported**. No dataset is dropped after its results are seen.
- A dataset is excluded only for a technical reason stated before its full results are inspected (for example, a failed download, or ERM worst-group accuracy of 0 on every seed so that the endpoint is degenerate). Any exclusion is reported.

## C. Scoring the diagnostic
- The main claim about the diagnostic is its predictive accuracy across all pre-registered and screened datasets: the number of datasets where the prediction (help / no help) matched the primary test outcome, out of the total.
- It is reported with an exact binomial test against 50% and a 2×2 table.
- Waterbirds and the datasets already in the manuscript are shown separately, since the rule was fitted on them.

## D. Robustness of the diagnostic
- The diagnostic is recomputed on the **validation** split as well as the test split, from the same ERM checkpoint.
- The prediction uses the test-split value, as registered.
- Any disagreement between the validation and test values is reported.

## E. Comparison baselines (reviewer item 4)
- On every pre-registered and screened dataset, the following label-using references are computed post hoc from the ERM feature dumps, so that the gap to methods using group labels is stated for each dataset:
  - DFR, last-layer retraining with validation group labels;
  - ERM + logit adjustment.
- The paper positions SDrop against methods that use **no group labels during training**. It does not claim to match methods that do.


ewpage

