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
