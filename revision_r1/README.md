# Neurocomputing revision R1 (NEUCOM-D-26-06952) — code, queues, results

Frozen snapshot of the code version used for every experiment reported in the revised manuscript
"Suppressive Dropout: Dropping Dominant Channels Helps Under Data Scarcity and Spurious Correlation" (October 2026).

- `code/`: training and evaluation code as run on the GPU pods (`train.py` with `--gamma`, `--delta`, `--method targeted`, `maxdrop`, `maxdrop_ch`, `--select topk`, `--norm mean` default; `sdrop.py`, `baselines.py`, `model.py`, `dataset.py`, `evaluate.py`, analysis helpers).
- `queues/`: the exact run queues (`run_*.sh`) and pod entry scripts; `_queue_lib.sh` is the helper every queue sources. Each `run` line gives the full `train.py` command for one reported run.
- `scripts/`: `neucom_run_analysis.py` (Part I tables, paired tests, BH over the stated family), `plot_memorisation_delay.py`, `bh_fdr.py` (test-ID table), `check_la_validation.py` (validation-tuned logit-adjustment check), pod and volume helpers.
- `results/cifar100/`, `results/waterbirds/`: per-seed `*_history.csv` for every run used in the tables (one row per epoch; last row of a Waterbirds history repeats the best-validation checkpoint's test metrics). The `ece` column is from a mis-scaled instrument and is not used in the paper.
- `results/results_RUN_2026-10-06.md`, `results_overhead_2026-10-06.md`, `memdelay_numbers.txt`: the generated tables the manuscript copies from.
- `registration/`: the registration documents (predictions and significance bars fixed before the corresponding runs).

Reproduce a table: find its runs in `queues/`, run those `train.py` commands with the listed seeds, then `python scripts/neucom_run_analysis.py` (Part I) or `python scripts/bh_fdr.py` (Part II–III).
