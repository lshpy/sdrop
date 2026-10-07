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
