# Screening datasets (fixed 2026-09-29, before any SDrop run on them)

This file extends `PREREG_DATASETS.md` (cct20, isic_artifact). The aim is to collect as many public datasets as we can that have a known shortcut and a per-image confound label. On each one we run ERM and the backbone diagnostic (`analyze_units.py`), write the prediction down, and then run the full experiment on all of them and report every result.

Every dataset is stored as Waterbirds-format parquet (`image{bytes}`, `label`, `place`, with `place` = the confound), or it reuses the CelebA parquet the volume already has. `train.py`, `evaluate_groups.py` and `analyze_units.py` therefore run on them unchanged. The same rules as `PREREG_DATASETS.md` apply: one ImageNet ResNet-50, checkpoint chosen on validation, test used once.

| name | domain / shortcut | classes | confound (`place`) | train / val / test | download | license |
|---|---|---|---|---|---|---|
| `in9bg` | ImageNet objects pasted on backgrounds: background class | 9 | IN-9 class of the background (9) | 2,700 / 450 / 8,100 | 280 MB | GitHub repo has no license file; images are ImageNet (research use only) |
| `counteranimal` | real iNaturalist animal photos: typical vs atypical background | 45 | 0 common / 1 counter background | ≈4,700 / ≈2,100 / ≈6,500 (see below) | 1.8 GB | none on the HF card; cite the paper |
| `fairface_age` | faces, BFFHQ-style: age target, gender shortcut | 2 | gender | 18,948 / 800 / 2,000 | 0.56 GB | CC BY 4.0 |
| `celeba_wavy` | CelebA Wavy_Hair, gender shortcut | 2 | Male | official 162,770 / 19,867 / 19,962 | 0 (on volume) or 1.4 GB | CelebA research-only; HF copy MIT-tagged |
| `celeba_glasses` | CelebA Eyeglasses, gender shortcut | 2 | Male | same | same | same |

Pipeline: run `bash pod_entry_screen_diag.sh`. It preps everything, marking each dataset with a `data/<name>/.done` file. Only `SHARD=0` downloads; other shards wait up to `PREP_WAIT_MIN`. It then calls `run_screen_diag.sh`, which runs one ERM seed 0 per dataset followed by `analyze_units.py`, writing to `checkpoints_screen/screen_units.csv`.

---

## in9bg: ImageNet-9 Backgrounds, re-composited

- **Source.** Xiao, Engstrom, Ilyas & Madry, *Noise or Signal: The Role of Image Backgrounds in Object Recognition*, ICLR 2021.
  - Release: `https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz` (279.6 MB, no login).
  - Contents: 9 super-classes × 450 images at 224 px. Each image has `original`, `fg_mask/*.npy` (boolean object mask) and `only_bg_t` (the background with the object region filled by tiling), plus the fixed `mixed_same`/`mixed_rand`/`mixed_next` composites.
  - Only these evaluation images are released. The IN-9 training set needs ImageNet itself.
- **Why re-composite instead of using mixed_same / mixed_rand?** The released composites have no training split and no correlated training distribution. The masks and tiled backgrounds let us build a Waterbirds-style benchmark with an exact per-image background label.
- **Construction (`prep_in9bg.py`, seed 0).**
  - Splits: per class, a seeded permutation splits the 450 objects 300 / 50 / 100 into train / val / test. The 450 backgrounds are split the same way, independently, because `only_bg_t` is a partly different image set: only 1,516 of its 4,050 ids are also objects. No background image is shared between splits.
  - Composite = `mask·original_i + (1−mask)·only_bg_t_j`, never using the object's own photo. Resized to 256 px and saved as JPEG q90.
  - `label` = object class. `place` = class of the background image.
  - **Train:** place = label with probability **ρ = 0.9**, otherwise one of the other 8 classes uniformly. Realized P(place = label) = 0.904. Off-diagonal cells hold 0–11 images (mean ≈ 3.8).
  - **Why ρ = 0.9 and not Waterbirds' 0.95?** The confound has 9 values, so the minority mass is spread over 8 off-diagonal cells per class. At 0.95 the mean off-diagonal cell has ≈ 1.9 images, and a third of them would be empty. 0.9 keeps the same kind of regime: a strongly predictive background with rare, but present, conflicts.
  - **Validation:** each object gets one background of a uniformly random class, so place is independent of label.
  - **Test:** each of the 100 test objects per class is pasted onto one random background of *each* of the 9 classes. That gives 81 cells of exactly 100 images; the 72 cells with place ≠ label are bias-conflicting.
- **Primary metric:** worst-group accuracy over the 81 cells (`test_worst_group`). Every cell has n = 100, so one image is 1 pp and the minimum is not driven by a tiny cell.
  - Secondary: the paper's BG-gap = accuracy on place = label cells − mean accuracy on place ≠ label cells (from the dump).
- **Unusual.**
  - The images are ImageNet *training* images (ids like `n02085620_16757`), so the pretrained backbone has seen every object and background. This is shared by all arms and does not bias the SDrop-vs-random comparison, but it must be stated.
  - Composites have mask-edge artifacts, like Waterbirds.
- Checked locally on the full archive: 14 s CPU, 196 MB output. Train cell table and 81×100 test cells verified.

## counteranimal: CounterAnimal (real photos, atypical backgrounds)

- **Source.** Wang et al., *A Sober Look at the Robustness of CLIPs to Spurious Features*, NeurIPS 2024.
  - Data: HF `TMLR-Group-HF/counteranimal`, 4 parquet files, 1.83 GB, not gated, pinned to commit `e7811153bef8cf5aa274e22b9db4bf002156e376`.
  - 13,334 iNaturalist photos of 45 ImageNet animal classes. Each photo has a background tag (34 values such as `common-snow`, `counter-grass`), assigned per class by the authors.
  - Overall 7,408 common / 5,926 counter.
  - Released as a single test split.
- **License:** the HF card declares none; cite the paper. iNaturalist photos carry their own (mostly CC) licenses.
- **Construction (`prep_counteranimal.py`, seed 0).**
  - `label` = class index 0–44 in ImageNet-id order. Columns `imagenet_idx` and the full `type` string are kept.
  - `place` = 0 common / 1 counter.
  - Per class, common images are split 60 / 10 / 30 % into train / val / test.
  - Counter images: `round(n_train_common · 0.05/0.95)` go to train, so **P(common | train) = 0.95**, as in Waterbirds. The rest are split 1 : 3 between val and test.
  - Prep aborts unless all 45 classes have both backgrounds in every split. The val/test AUC in `train.py` needs every class present.
  - **Why construct splits?** There is no official training split. The benchmark's premise is that the counter background is rare in web data, and a 95 / 5 train mix reproduces that.
- **Primary metric:** class-balanced accuracy on the **counter** cells = mean over the 45 `(class, counter)` cells.
  - Not worst-group: 90 cells of roughly 20–180 images mean the minimum is set by the hardest *class*, not by the shortcut. The paper's own metric is the common−counter accuracy drop.
  - Secondary: `test_worst_group`, and the common−counter gap (class-balanced).
- **Unusual.**
  - The common/counter split was defined from *CLIP* failures, so it is selected on a different model. It is a real, if model-dependent, shortcut.
  - "Counter" means a different background for each class (snow for one class, grass for another), so `place` is binary "atypical background", not a single background type. The diagnostic's background selectivity is one-vs-rest over this binary label.
  - All 45 classes are ImageNet classes; photos come from iNaturalist, not ImageNet.
- Local check: one shard (12 classes) prepped (splits, cells, P(counter | train) = 0.046). The full 4-shard check is reported in the hand-off notes.

## fairface_age: BFFHQ-style age/gender on FairFace

- **Source.** Kärkkäinen & Joo, *FairFace*, WACV 2021.
  - Data: HF `HuggingFaceM4/FairFace`, config `0.25` (padding 0.25, 224 px), 3 parquet files, 0.56 GB, CC BY 4.0, pinned to commit `54d573cdb8b5af490ba8da9da2799628f6e5c496`.
  - Official train 86,744 / validation 10,954.
- **Why this instead of BFFHQ?** BFFHQ (Lee et al., NeurIPS 2021; target age, bias gender) is distributed only on Google Drive, and its SpuCo copy's download link is dead. FairFace has the same labels under a clear license, so we rebuild the BFFHQ task on it.
- **Construction (`prep_fairface_age.py`).**
  - `label`: 0 young (FairFace age 10–19, 20–29) / 1 old (40–49, 50–59). These are BFFHQ's bands; other ages are dropped.
  - `place` = gender (0 male, 1 female). Aligned cells follow BFFHQ: young/female and old/male.
  - **Train** (from official train): 9,000 per aligned cell and 474 per conflicting cell, so **P(aligned) = 0.95**.
    - The available old/male pool is ≈ 10.6k, estimated from one of the two shards, and it caps the majority size.
    - BFFHQ uses 99.5 %. We use Waterbirds' 95 % so every screening set shares the same regime and the minority cells are not near-empty.
  - **Val / test:** official validation, split 1/3 : 2/3 with its own seed. Each (label, gender) cell is cut to 200 (val) or 500 (test), so both are group-balanced. The smallest available cells are ≈ 270 and ≈ 550.
- **Primary metric:** worst-group accuracy over the 4 cells (`test_worst_group`), with n = 500 per cell.
- **Unusual.**
  - Race and the remaining ages are not controlled.
  - Age labels are FairFace's crowd estimates.
- Checked locally: prep on shard 0 + validation with small cell sizes, loader, 1-epoch train and dump.

## celeba_wavy / celeba_glasses: more CelebA target attributes, gender confound

- **Source.** The CelebA parquet already used for `celeba` (HF `tpremoli/CelebA-attrs`, commit `ed9021d2871ceddbd3cf0fb642544bd7c60c5152`, 40 attribute columns). Official splits, no new download if the volume has `data/celeba`.
- **Construction.** No prep. `get_celeba(label=<attr>, spurious='Male')`; the default is unchanged (Blond_Hair).
- **Pairs chosen.** From the test-set cell counts (target, Male), picking pairs with a strong gender correlation and a usable minority cell:

  | pair | test cells (y=0 f, y=0 m, y=1 f, y=1 m) | minority cell |
  |---|---|---|
  | Wavy_Hair × Male | 6,846 / 7,526 / 4,563 / **932** | wavy males |
  | Eyeglasses × Male | 11,166 / 7,318 / **243** / 1,140 | women with glasses |

- **Rejected pairs.**
  - Heavy_Makeup × Male and Wearing_Lipstick × Male: 9 and 32 minority test images, so the worst group is noise.
  - Attractive, Smiling and Young × Male: correlation too weak to be a shortcut.
  - Both chosen pairs are used in the spurious-correlation literature (e.g. Wavy_Hair and Heavy_Makeup/Male in several debiasing papers).
- **Protocol.** The paper's CelebA ERM, not the Waterbirds one: 8 epochs, lr 1e-3, wd 1e-4, bs 128 (`run_revision_celeba.sh`). Train has 163k images, and 30 epochs would cost ~4× more for no benefit.
- **Primary metric:** `test_worst_group` over the 4 cells.
- **Unusual.** These share images and the confound with `celeba`, so they are not independent datasets. They test whether the diagnostic's prediction moves with the *target* while the backbone input stays fixed.

---

## Investigated and rejected

| candidate | why not |
|---|---|
| SpuCoAnimals / SpuCoDogs / SpuCoBirds (Joshi et al. 2023) | SpuCoAnimals needs ImageNet. The Dogs/Birds tarballs (`ucla.box.com/shared/static/...` in `spuco`) return HTTP 404. |
| SpuCoSun | foregrounds are diffusion-generated and SUN397 is downloaded separately; not real photos |
| UrbanCars (Li et al. 2023) | must be generated from Stanford Cars + Places365 + LVIS, no released images |
| ImageNet-W, Hard ImageNet, Salient ImageNet, Spurious ImageNet | need ImageNet |
| FOCUS (Kattakinda & Feizi, ICML 2022) | good fit (real photos, per-image location/weather/time), but the only download is a Box share page that needs JavaScript (no direct file URL). Revisit if a direct link is found. |
| BFFHQ | Google Drive only; replaced by `fairface_age` |
| ISIC 2019 colored patches (Rieger et al. 2020) | patches occur only on benign images, so there is no bias-conflicting test cell. The `patches` artifact is already a column in `isic_artifact`. |
| Camelyon17-WILDS (hospital shortcut) | Available (HF `wltjr1007/Camelyon17-WILDS`, 10.7 GB, CC0 via WILDS), but label and hospital are not correlated in the benchmark. A constructed correlation would need a slide-level split, and per-slide counts are very uneven (slides 40–44 hold 247 patches in total). Also 96-px patches, a resolution confound. Low priority. |
| TerraIncognita, iWildCam | overlap `cct20` (same camera-trap location shortcut) |
| UTKFace | faces covered by `fairface_age` and CelebA pairs; non-commercial license |

## Diagnostic and prediction

The rule is the same as in `PREREG_DATASETS.md`. SDrop (energy, layer4) is predicted to beat random dropping on the primary metric iff layer4 `bg_ratio` > 1.2 and `cls_ratio` ≤ 1.0 (the rule registered in prereg/PREREG_field_datasets_2026-09-29.md and applied by the addendum, section B). `bg_ratio` is computed against `place` as listed above: a 9-way background class for in9bg, binary for the rest.

**Before any SDrop run on these sets, the authors still have to fix the numeric thresholds, the arms and the seed count.** The suggestion is the same as for MetaShift: ERM / Energy / Random / Shuffle, 6 seeds, p = 0.1, L4.

## Launch

```
# after syncing sdrop_pod_code to /workspace/sdrop; ~2.7 GB download (4.1 GB if CelebA is not on the volume)
bash pod_entry_screen_diag.sh                          # one pod, MAX_HOURS=4 default
SHARD=0 NSHARD=2 bash pod_entry_screen_diag.sh         # pod A (preps)
SHARD=1 NSHARD=2 bash pod_entry_screen_diag.sh         # pod B (waits for .done, PREP_WAIT_MIN=90)
```
