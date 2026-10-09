# Energy top-k (SDrop) vs max top-k (Max-Drop channel) at layer4 -- Waterbirds test, offline
Date 2026-10-08. k = 205 of 2048 channels (rate 0.1); ranking by mean-square energy is identical to ranking by the bounded transform 1-(1+gamma*Sigma)^-delta (monotone). Both checkpoints run in eval mode with no dropout.

## Checkpoint: erm
Accuracy avg 87.76, worst-group 58.57 g0_landbird_land 99.7, g1_landbird_water 82.4, g2_waterbird_land 58.6, g3_waterbird_water 94.1

### Overlap and removed mass (per-sample means)
| subset | Jaccard | inter/k | mass energy-rule | mass max-rule | mass random-k |
|---|---|---|---|---|---|
| overall | 0.624 | 0.768 | 0.447 | 0.421 | 0.100 |
| g0_landbird_land | 0.627 | 0.770 | 0.446 | 0.421 | 0.100 |
| g1_landbird_water | 0.622 | 0.766 | 0.438 | 0.412 | 0.100 |
| g2_waterbird_land | 0.634 | 0.775 | 0.457 | 0.431 | 0.100 |
| g3_waterbird_water | 0.614 | 0.760 | 0.471 | 0.440 | 0.100 |
| minority (g1+g2) | 0.625 | 0.768 | 0.442 | 0.416 | 0.100 |

### Channels by category (means over channel-sample pairs)
| subset | category | pairs | per sample | peakedness max/sum | mean-square | max | centre 3x3 frac | border ring frac |
|---|---|---|---|---|---|---|---|---|
| overall | max_only | 276070 | 47.6 | 0.146 | 1.432 | 4.343 | 0.366 | 0.296 |
| overall | energy_only | 276070 | 47.6 | 0.065 | 2.313 | 3.472 | 0.301 | 0.334 |
| overall | both | 911700 | 157.4 | 0.095 | 3.882 | 5.400 | 0.390 | 0.254 |
| minority | max_only | 137950 | 47.6 | 0.149 | 1.476 | 4.478 | 0.357 | 0.312 |
| minority | energy_only | 137950 | 47.6 | 0.067 | 2.389 | 3.569 | 0.304 | 0.333 |
| minority | both | 455935 | 157.4 | 0.097 | 3.999 | 5.553 | 0.391 | 0.256 |

### Selection coverage over the test set
| subset | rule | channels ever selected | entropy (nats, max 7.625) | effective #channels | top-20 share | selected in >50% of samples |
|---|---|---|---|---|---|---|
| overall | energy | 2034 | 7.360 | 1572 | 0.042 | 1 |
| overall | max | 2035 | 7.320 | 1510 | 0.041 | 3 |
Correlation of per-channel selection frequency between rules (overall): 0.938
| minority | energy | 2030 | 7.311 | 1497 | 0.050 | 10 |
| minority | max | 2030 | 7.283 | 1455 | 0.048 | 9 |
Correlation of per-channel selection frequency between rules (minority): 0.926

## Checkpoint: sdrop_topk
Accuracy avg 86.97, worst-group 71.65 g0_landbird_land 99.3, g1_landbird_water 76.8, g2_waterbird_land 71.7, g3_waterbird_water 94.7

### Overlap and removed mass (per-sample means)
| subset | Jaccard | inter/k | mass energy-rule | mass max-rule | mass random-k |
|---|---|---|---|---|---|
| overall | 0.578 | 0.730 | 0.419 | 0.387 | 0.100 |
| g0_landbird_land | 0.614 | 0.760 | 0.430 | 0.404 | 0.100 |
| g1_landbird_water | 0.576 | 0.728 | 0.405 | 0.374 | 0.100 |
| g2_waterbird_land | 0.556 | 0.713 | 0.425 | 0.390 | 0.100 |
| g3_waterbird_water | 0.483 | 0.649 | 0.420 | 0.370 | 0.101 |
| minority (g1+g2) | 0.572 | 0.725 | 0.409 | 0.378 | 0.100 |

### Channels by category (means over channel-sample pairs)
| subset | category | pairs | per sample | peakedness max/sum | mean-square | max | centre 3x3 frac | border ring frac |
|---|---|---|---|---|---|---|---|---|
| overall | max_only | 320586 | 55.3 | 0.153 | 1.283 | 4.228 | 0.340 | 0.345 |
| overall | energy_only | 320586 | 55.3 | 0.064 | 2.108 | 3.306 | 0.291 | 0.349 |
| overall | both | 867184 | 149.7 | 0.097 | 3.389 | 5.129 | 0.382 | 0.273 |
| minority | max_only | 163312 | 56.4 | 0.158 | 1.296 | 4.368 | 0.325 | 0.375 |
| minority | energy_only | 163312 | 56.4 | 0.066 | 2.149 | 3.403 | 0.292 | 0.354 |
| minority | both | 430573 | 148.6 | 0.100 | 3.434 | 5.274 | 0.387 | 0.279 |

### Selection coverage over the test set
| subset | rule | channels ever selected | entropy (nats, max 7.625) | effective #channels | top-20 share | selected in >50% of samples |
|---|---|---|---|---|---|---|
| overall | energy | 2039 | 7.330 | 1526 | 0.044 | 6 |
| overall | max | 2039 | 7.269 | 1435 | 0.044 | 4 |
Correlation of per-channel selection frequency between rules (overall): 0.905
| minority | energy | 2034 | 7.260 | 1422 | 0.053 | 11 |
| minority | max | 2032 | 7.218 | 1364 | 0.050 | 10 |
Correlation of per-channel selection frequency between rules (minority): 0.874

Centre/border fractions are a crude spatial proxy on the 7x7 map, not a measurement of object or background content.

## Reading

The two rules pick largely the same channels: Jaccard 0.62 (ERM) and 0.58 (SDrop), i.e. 77% and 73% of the k=205 channels coincide; the minority groups give 0.62 and 0.57. The energy rule removes 45% / 42% of the squared-activation mass (ERM / SDrop), the max rule 42% / 39%, against 10% for a random k-subset. Channels chosen only by the max rule are spatially peaked (max/sum 0.15 / 0.15) with low mean-square energy (1.432 / 1.283), whereas channels chosen only by the energy rule are diffuse (max/sum 0.07 / 0.06) with higher mean-square energy (2.313 / 2.108) but a lower maximum (3.47 / 3.31 vs 4.34 / 4.23); their centre-3x3 mass fractions are 0.30 / 0.29 (energy-only) against 0.37 / 0.34 (max-only), a spatial proxy only. Over the test set the energy rule ever selects 2034 / 2039 channels (entropy 7.36 / 7.33 nats) and the max rule 2035 / 2039 (entropy 7.32 / 7.27). These are descriptive statistics of which channels each rule removes at test time; they do not establish why the two training arms differ.
