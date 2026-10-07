# Overhead measurements (R3-10)

## Peak GPU memory and epoch time, CIFAR-100 100% data, ResNet-18, batch 128, AMP, RTX PRO 6000 Blackwell Server Edition (2026-10-06, pod neucom6-mem, 2 epochs per arm, nvidia-smi sampled at 1 Hz)
| arm | peak memory (MiB) | epoch time (s) |
|---|---|---|
| no dropout | 1409 | 3.7 |
| SpatialDropout | 1411 | 3.7 |
| random channel dropout | 1411 | 3.9 |
| SDrop | 1411 | 4.0 |
| DropBlock | 1411 | 3.7 |
| Targeted Dropout | 1411 | 4.1 |
Memory overhead +2 MiB (one channel mask); time overhead SDrop +8% per epoch on this GPU; inference cost zero (training-only module, removed at eval).

## Per-epoch median wall time from the original (April) runs, single GPU, 120-epoch schedule
CIFAR-100: none 4.94 s, SpatialDropout 4.71, SDrop 4.90, SGridLC G=3 5.79, G=4 5.80, G=3+4 6.57.
SVHN: none 7.03, SpatialDropout 7.17, SDrop 7.14, SGridLC G=3 8.86, G=4 8.62, G=3+4 10.12.
