# SOP results: ANTROUTE vs Cheng (2023)

Generated 2026-10-18T06:36:54+00:00 · 10 trips × 5 scenarios × 2 seeds · network built 2026-10-18T04:21:15.615515+00:00

Path cost is the CRITIC ten-feature cost of each road when entered, for both algorithms. Convergence rate = iterations until the initial search first found its final best route. Traffic is modeled, not measured. Lower is better for every measure.

## SOP 1. Routing performance under selected traffic conditions

### Distance travelled (km)

| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 8.83 ± 3.31 (n=20) | 10.47 ± 4.25 (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 8.83 ± 3.39 (n=20) | 13.15 ± 7.35 (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 8.84 ± 3.26 (n=20) | 11.29 ± 5.30 (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 9.00 ± 3.22 (n=20) | 12.46 ± 6.16 (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 8.98 ± 3.47 (n=20) | 12.04 ± 6.35 (n=20) |
| All conditions | 8.89 ± 3.26 (n=100) | 11.88 ± 5.93 (n=100) |

### Path cost

| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 28.70 ± 9.06 (n=20) | 42.40 ± 13.51 (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 35.59 ± 14.77 (n=20) | 76.81 ± 44.00 (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 30.93 ± 12.25 (n=20) | 55.09 ± 28.37 (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 47.53 ± 15.21 (n=20) | 77.76 ± 28.36 (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 48.34 ± 16.35 (n=20) | 80.83 ± 32.47 (n=20) |
| All conditions | 38.22 ± 15.84 (n=100) | 66.58 ± 33.91 (n=100) |

## SOP 2. Computational performance

### Computational time (s)

| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 3.13 ± 2.57 (n=20) | 6.72 ± 4.85 (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 4.06 ± 2.73 (n=20) | 13.78 ± 7.57 (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 4.45 ± 2.27 (n=20) | 12.94 ± 5.77 (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 6.85 ± 8.30 (n=20) | 14.46 ± 9.23 (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 5.20 ± 3.25 (n=20) | 11.73 ± 4.70 (n=20) |
| All conditions | 4.74 ± 4.53 (n=100) | 11.93 ± 7.08 (n=100) |

### Initial routing time (s)

| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 3.13 ± 2.57 (n=20) | 6.72 ± 4.85 (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 2.37 ± 1.77 (n=20) | 7.76 ± 4.19 (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 2.64 ± 1.05 (n=20) | 7.78 ± 3.67 (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 2.46 ± 1.13 (n=20) | 7.66 ± 4.49 (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 2.57 ± 1.60 (n=20) | 6.27 ± 2.98 (n=20) |
| All conditions | 2.64 ± 1.70 (n=100) | 7.24 ± 4.05 (n=100) |

### Dynamic re-routing time (per re-route) (s)

| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | – | – |
| Morning peak begins during the trip (Tue 06:50) | 1.99 ± 1.53 (n=17) | 5.46 ± 2.59 (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 2.01 ± 1.48 (n=18) | 5.16 ± 2.59 (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 4.38 ± 7.66 (n=20) | 5.79 ± 2.80 (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 2.63 ± 1.83 (n=20) | 5.06 ± 2.47 (n=20) |
| All conditions | 2.80 ± 4.24 (n=75) | 5.37 ± 2.58 (n=80) |

### Dynamic re-routing time (per trip) (s)

| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 0.00 ± 0.00 (n=20) | 0.00 ± 0.00 (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 1.69 ± 1.58 (n=20) | 6.02 ± 4.45 (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 1.81 ± 1.53 (n=20) | 5.16 ± 2.59 (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 4.38 ± 7.66 (n=20) | 6.79 ± 5.16 (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 2.63 ± 1.83 (n=20) | 5.47 ± 2.64 (n=20) |
| All conditions | 2.10 ± 3.86 (n=100) | 4.69 ± 4.17 (n=100) |

### Re-routes per trip

| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 0.00 ± 0.00 (n=20) | 0.00 ± 0.00 (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 0.85 ± 0.37 (n=20) | 1.05 ± 0.22 (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 0.90 ± 0.31 (n=20) | 1.00 ± 0.00 (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 1.00 ± 0.00 (n=20) | 1.10 ± 0.31 (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 1.00 ± 0.00 (n=20) | 1.10 ± 0.31 (n=20) |
| All conditions | 0.75 ± 0.44 (n=100) | 0.85 ± 0.48 (n=100) |

### Convergence rate: iterations to convergence (initial search) (iterations)

| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 5.35 ± 3.12 (n=20) | 5.70 ± 2.54 (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 4.65 ± 3.05 (n=20) | 4.65 ± 2.18 (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 4.90 ± 3.13 (n=20) | 6.10 ± 2.20 (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 5.90 ± 2.99 (n=20) | 4.85 ± 1.95 (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 4.80 ± 3.30 (n=20) | 4.85 ± 1.95 (n=20) |
| All conditions | 5.12 ± 3.09 (n=100) | 5.23 ± 2.21 (n=100) |

### Iterations to convergence (all searches) (iterations)

| Traffic condition | ANTROUTE mean ± SD | Cheng mean ± SD |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 5.35 ± 3.12 (n=20) | 5.70 ± 2.54 (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 3.88 ± 2.18 (n=20) | 2.84 ± 1.10 (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 4.00 ± 3.20 (n=20) | 3.55 ± 1.10 (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 5.22 ± 1.86 (n=20) | 2.95 ± 1.07 (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 4.70 ± 2.64 (n=20) | 2.88 ± 1.01 (n=20) |
| All conditions | 4.63 ± 2.67 (n=100) | 3.58 ± 1.82 (n=100) |

## SOP 3. Significant difference between ANTROUTE and the baseline

Paired by trip and traffic condition (seeds averaged). Difference = ANTROUTE − Cheng; negative favours ANTROUTE for every measure. α = 0.05.

### All conditions

| Measure | n | ANTROUTE | Cheng | Mean difference | Test | Statistic | p | Significant |
|---|---|---|---|---|---|---|---|---|
| Distance travelled | 50 | 8.89 | 11.88 | -2.99 | Wilcoxon signed-rank | W = 9.0 | < 0.0001 | **Yes** |
| Path cost | 50 | 38.22 | 66.58 | -28.36 | Wilcoxon signed-rank | W = 0.0 | < 0.0001 | **Yes** |
| Computational time | 50 | 4.74 | 11.93 | -7.19 | Wilcoxon signed-rank | W = 48.0 | < 0.0001 | **Yes** |
| Convergence rate | 50 | 5.12 | 5.23 | -0.11 | Wilcoxon signed-rank | W = 568.0 | 0.5086 | No |

### Off-peak, no change (Tue 10:10, historical weather)

| Measure | n | ANTROUTE | Cheng | Mean difference | Test | Statistic | p | Significant |
|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 8.83 | 10.47 | -1.64 | paired t-test | t(9) = -4.24 | 0.0022 | **Yes** |
| Path cost | 10 | 28.70 | 42.40 | -13.70 | paired t-test | t(9) = -6.87 | < 0.0001 | **Yes** |
| Computational time | 10 | 3.13 | 6.72 | -3.59 | paired t-test | t(9) = -2.66 | 0.0261 | **Yes** |
| Convergence rate | 10 | 5.35 | 5.70 | -0.35 | paired t-test | t(9) = -0.35 | 0.7330 | No |

### Morning peak begins during the trip (Tue 06:50)

| Measure | n | ANTROUTE | Cheng | Mean difference | Test | Statistic | p | Significant |
|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 8.83 | 13.15 | -4.32 | Wilcoxon signed-rank | W = 0.0 | 0.0020 | **Yes** |
| Path cost | 10 | 35.59 | 76.81 | -41.22 | Wilcoxon signed-rank | W = 0.0 | 0.0020 | **Yes** |
| Computational time | 10 | 4.06 | 13.78 | -9.73 | Wilcoxon signed-rank | W = 0.0 | 0.0020 | **Yes** |
| Convergence rate | 10 | 4.65 | 4.65 | 0.00 | paired t-test | t(9) = 0.00 | 1.0000 | No |

### Evening peak begins during the trip (Tue 16:50)

| Measure | n | ANTROUTE | Cheng | Mean difference | Test | Statistic | p | Significant |
|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 8.84 | 11.29 | -2.45 | paired t-test | t(9) = -3.51 | 0.0066 | **Yes** |
| Path cost | 10 | 30.93 | 55.09 | -24.16 | paired t-test | t(9) = -4.39 | 0.0017 | **Yes** |
| Computational time | 10 | 4.45 | 12.94 | -8.49 | Wilcoxon signed-rank | W = 0.0 | 0.0020 | **Yes** |
| Convergence rate | 10 | 4.90 | 6.10 | -1.20 | paired t-test | t(9) = -1.12 | 0.2901 | No |

### Road closure 2 min into the trip (Tue 08:10)

| Measure | n | ANTROUTE | Cheng | Mean difference | Test | Statistic | p | Significant |
|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 9.00 | 12.46 | -3.45 | Wilcoxon signed-rank | W = 1.0 | 0.0039 | **Yes** |
| Path cost | 10 | 47.53 | 77.76 | -30.23 | paired t-test | t(9) = -4.92 | 0.0008 | **Yes** |
| Computational time | 10 | 6.85 | 14.46 | -7.61 | paired t-test | t(9) = -2.37 | 0.0420 | **Yes** |
| Convergence rate | 10 | 5.90 | 4.85 | 1.05 | paired t-test | t(9) = 0.96 | 0.3612 | No |

### Clear to heavy rain 3 min into the trip (Tue 08:10)

| Measure | n | ANTROUTE | Cheng | Mean difference | Test | Statistic | p | Significant |
|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 8.98 | 12.04 | -3.07 | Wilcoxon signed-rank | W = 1.0 | 0.0039 | **Yes** |
| Path cost | 10 | 48.34 | 80.83 | -32.50 | paired t-test | t(9) = -4.82 | 0.0009 | **Yes** |
| Computational time | 10 | 5.20 | 11.73 | -6.53 | paired t-test | t(9) = -10.65 | < 0.0001 | **Yes** |
| Convergence rate | 10 | 4.80 | 4.85 | -0.05 | paired t-test | t(9) = -0.04 | 0.9683 | No |

## Runs that failed

None.
