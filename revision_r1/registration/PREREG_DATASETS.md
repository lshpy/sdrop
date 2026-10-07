# Pre-registered field datasets (fixed 2026-09-29, before any SDrop run on them)

Two real-world ("field") datasets with a known non-object confound, picked before any result.
Both are converted to the Waterbirds parquet format (`image{bytes}`, `label`, `place`), so `train.py`,
`evaluate_groups.py` and `analyze_units.py` run on them unchanged. `place` is the confound.

| | `cct20` | `isic_artifact` |
|---|---|---|
| Domain | camera traps, location/background shortcut | dermoscopy, ruler-artifact shortcut |
| Source | Beery et al., *Recognition in Terra Incognita*, ECCV 2018 (CCT20 subset of Caltech Camera Traps), via LILA | Bissoto et al., *Debiasing Skin Lesion Datasets and Models? Not So Fast*, CVPRW 2020 (artifact labels + trap splits) on ISIC 2018 Task 1/2 images |
| Images URL | `https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz` (6.49 GB; Azure mirror `https://lilawildlife.blob.core.windows.net/lila-wildlife/caltechcameratraps/...`) | `https://isic-archive.s3.amazonaws.com/images/<ISIC_id>.jpg`, one per image (2,594 images, 1.55 GB) |
| Labels URL | `.../caltechcameratraps/eccv_18_annotations.tar.gz` (3 MB) | `https://raw.githubusercontent.com/alceubissoto/debiasing-skin/f93c8654f50c581544f32ad6b1450cb925a176ff/trap-sets/isic_annotated_{train,val,test}1.csv` (commit pinned) |
| License | CDLA-Permissive-1.0 (LILA) | ISIC images: CC-BY-NC 4.0 (challenge release; archive lists some as CC-0). Bissoto CSVs: the repo has no license file, so cite the paper |
| Login | none | none |
| Prep | `prep_cct20.py` (streams the tar, no extraction) | `prep_isic_artifact.py` (32 parallel fetches, 5 retries) |
| Stored as | 256x256 JPEG q90 (the first step of `waterbirds_transforms`); same transform as Waterbirds | same |
| Classes | 12: bird, bobcat, car, cat, coyote, dog, opossum, rabbit, raccoon, rodent, skunk, squirrel | 2: 0 benign, 1 malignant (melanoma) |
| Confound (`place`) | camera location id (10 train locations; 9 more unseen in test) | ruler visible (0/1), from Bissoto's manual labels |
| Train | official `train`: 13,496 | trap `train1` minus `val1`: 1,563 |
| Validation (checkpoint selection) | official `cis_val`: 2,485 | trap `val1`: 259 |
| Test | `cis_test` 14,274 + `trans_test` 21,252 = 35,526 | trap `test1`: 772 (the same in all five trap splits) |
| Architecture | ImageNet ResNet-50, Waterbirds ERM protocol (30 ep, SGD lr 1e-3, momentum 0.9, wd 1e-4, bs 64) | same |

## Construction rules (fixed before download; the code implements exactly these)

**cct20**
- Keep images with exactly one annotated category (multi-species frames dropped).
- Keep the categories with at least 100 such train images. That drops `empty` (no animal), badger (9), fox (5) and deer (45).
- Test = `cis_test` ∪ `trans_test`, restricted to (class, location) cells with at least 20 images. The count is taken on the annotations, so the rule depends neither on the download nor on any model. 117 of 166 cells are kept (57 cis, 60 trans; smallest cell 21). Column `split` = `cis`/`trans`. `trans_val` (one location) is unused.
- The shortcut in training: every class sits mostly at a few locations. The top location holds 26–100 % of a class (car 100 %, rodent 83 %, cat 74 %, opossum 63 %).

**isic_artifact**
- `place` = `ruler`. It is the artifact most correlated with the label: phi = +0.41 in train1, −0.67 in test1. The other seven artifact columns (dark_corner, hair, gel_border, gel_bubble, ink, patches, vasc) are kept in the parquet.
- Train cells (label, ruler): benign/no 867, benign/ruler 398, malignant/no 58, malignant/ruler 240. So P(ruler | malignant) = 0.81 and P(ruler | benign) = 0.31.
- Test cells: benign/no 89, **benign/ruler 529**, **malignant/no 141**, malignant/ruler 13. The two bold cells conflict with the training correlation.
- Trap split 1 only. Splits 1–5 differ only in which 259 train images form `val`.

## Groups and metrics

`train.py` records `test_worst_group` / `test_avg_group` over every (label, place) cell in the test parquet, and `--dump_feats` saves `test_y`, `test_places` and `test_logits` for the rest. For cct20 the history keys are `test_acc_g<class>_<location>` (the `_` separator appears only when some id is 10 or more, so existing benchmarks keep their keys).

- **cct20, primary:** worst-group accuracy over the 117 retained (class, location) cells (= `test_worst_group`).
  - Secondary: accuracy on `trans` images (unseen locations), and on `cis`.
  - Secondary: mean accuracy over the 18 cis cells whose class had less than 5 % of its training images at that location (2,479 images). These location-minority cells are the Waterbirds analogue.
  - All secondaries are computed from the dump. The trans locations are {0, 7, 28, 40, 46, 78, 100, 105, 130}.
- **isic_artifact, primary:** worst bias-conflicting group = min(acc benign/ruler, acc malignant/no-ruler) = min(`test_acc_g01`, `test_acc_g10`).
  - The 4-cell `test_worst_group` is also reported, but its malignant/ruler cell has only 13 images (one image = 7.7 pp), so it is not the primary metric.
  - Secondary: test AUC (Bissoto et al.'s metric), from the dump logits.

## Diagnostic (run first, on ERM only)

`run_prereg_diag.sh`: one ERM seed (seed 0) per dataset, then `analyze_units.py --dataset <name>` on the best checkpoint. It writes the per-site `bg_ratio` / `cls_ratio` (ResNet layer1–4) to `checkpoints_prereg/prereg_units.csv`.
- Selectivity is taken over `place` as for Waterbirds. For cct20 that is one-vs-rest over the 19 location ids; for isic_artifact it is over the binary ruler label.
- The test set is used, as in Table 7.
- The prediction to register comes from the `analyze_units.py` docstring: SDrop (energy, layer4) should beat random and shuffled dropping on the primary metric only if layer4 `bg_ratio` > 1 and `cls_ratio` is not > 1. **The authors still have to fix the numeric thresholds, the arms and the seed count before looking at any SDrop run.** A suggestion matching MetaShift: ERM / Energy / Random / Shuffle, 6 seeds, p = 0.1, L4.

## Not implemented

iWildCam (WILDS) was the fallback for camera traps. It is not needed now that CCT20 works, and it was not checked.

## Launch

```
# after syncing sdrop_pod_code to /workspace/sdrop; ~8 GB download and < 1 GPU-hour expected
bash pod_entry_prereg_diag.sh      # MAX_HOURS=3 default, self-terminates
```
