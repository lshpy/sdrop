# Pre-registration: independent replication of the object-size effect (new seeds only)

Frozen 2026-09-29, after the object-size experiment (PREREG_objsize_2026-09-29.md) and before any replication run. Author: Seunghyun Lee.

**Motivation.**
- In the original experiment the registered primary test (×0.3, Energy − Random) came out at +2.0, 6/6 seeds, p = 0.058. It narrowly missed the registered bar, and it is reported as a miss.
- The strongest secondary test was ×0.5, Energy top-k − Shuffle top-k: +6.0, 6/6, p = 0.005.
- This replication uses **new seeds only** and does not pool them with seeds 0–5, so it is an independent test rather than optional stopping.

**Data.** Unchanged: `data/in9bg_s03` and `data/in9bg_s05`, built by `prep_in9bg.py --scale 0.3 / 0.5`, the same parquet files as the original experiment.

**Design.** Same protocol as the original: ResNet-50, 30 epochs, SGD lr 1e-3, wd 1e-4, batch 64, L4, p = 0.1, `--dump_feats`.
- ×0.3: ERM, Energy, Random; seeds 6–17 (12 seeds).
- ×0.5: Energy top-k, Shuffle top-k; seeds 6–17 (12 seeds).
- 60 runs (`checkpoints_in9rep`). No tuning; no seed or arm is dropped.

**Tests.** Both are confirmatory; BH is applied within this pair, and both also join the paper's confirmatory family.
- R1: ×0.3 Energy − Random, worst-group accuracy, seed-paired two-sided t-test on seeds 6–17 only (n = 12). Success = mean > 0 and p < 0.05.
- R2: ×0.5 Energy top-k − Shuffle top-k, same test on seeds 6–17 (n = 12).

**Exploratory, reported whatever the outcome.**
- Energy − ERM at ×0.3.
- Pooled estimates over seeds 0–17, labelled as pooled.

All outcomes are reported.
