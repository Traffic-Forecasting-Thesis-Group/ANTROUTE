# SOP results: ANTROUTE vs Cheng (2023)

Generated 2026-10-20T07:36:04+00:00 · 10 trips × 5 scenarios × 2 seeds · network built 2026-10-19T13:40:54.159873+00:00

Path cost is the CRITIC ten-feature cost of each road when entered, for both algorithms. Convergence rate = N_conv, the iteration at which the initial search's best path cost had not improved for 20 consecutive iterations (Aadil et al., 2016); both algorithms stop on this rule, and a run that reaches the iteration limit first is recorded as not converged. Traffic is modeled, not measured. Lower is better for every measure.

## SOP 1. Routing performance under selected traffic conditions

### Distance travelled (km)

| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 8.89 ± 3.38 [4.62–15.63] (n=20) | 11.12 ± 5.47 [5.14–25.45] (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 8.86 ± 3.44 [4.57–15.63] (n=20) | 11.03 ± 4.67 [4.74–21.64] (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 8.97 ± 3.34 [4.62–15.63] (n=20) | 11.66 ± 6.00 [4.53–24.74] (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 9.54 ± 4.15 [4.90–19.33] (n=20) | 14.60 ± 7.00 [5.91–32.71] (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 9.03 ± 3.46 [4.64–15.67] (n=20) | 13.27 ± 7.14 [4.54–32.71] (n=20) |
| All conditions | 9.06 ± 3.50 [4.57–19.33] (n=100) | 12.34 ± 6.16 [4.53–32.71] (n=100) |

### Path cost

| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 32.73 ± 9.64 [19.39–51.69] (n=20) | 51.76 ± 19.78 [28.44–101.84] (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 33.84 ± 11.56 [19.16–58.22] (n=20) | 52.90 ± 19.09 [26.79–89.89] (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 33.41 ± 11.46 [19.00–56.20] (n=20) | 53.25 ± 24.80 [24.19–107.87] (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 39.01 ± 13.45 [21.49–64.17] (n=20) | 77.17 ± 30.83 [28.88–135.78] (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 38.06 ± 12.72 [20.95–59.76] (n=20) | 75.15 ± 33.86 [28.25–145.36] (n=20) |
| All conditions | 35.41 ± 11.88 [19.00–64.17] (n=100) | 62.05 ± 28.29 [24.19–145.36] (n=100) |

## SOP 2. Computational performance

### Computational time (s)

| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 2.01 ± 0.72 [0.68–3.76] (n=20) | 2.58 ± 1.04 [1.17–5.37] (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 3.17 ± 1.74 [0.66–7.27] (n=20) | 3.60 ± 1.16 [1.18–5.80] (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 3.26 ± 1.72 [0.62–7.67] (n=20) | 3.70 ± 1.18 [1.84–6.20] (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 5.18 ± 4.40 [1.20–17.22] (n=20) | 4.03 ± 1.68 [1.98–8.57] (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 3.79 ± 1.63 [1.70–7.90] (n=20) | 4.24 ± 1.85 [1.95–8.17] (n=20) |
| All conditions | 3.48 ± 2.56 [0.62–17.22] (n=100) | 3.63 ± 1.50 [1.17–8.57] (n=100) |

### Initial routing time (s)

| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 2.01 ± 0.72 [0.68–3.76] (n=20) | 2.45 ± 0.79 [1.17–4.00] (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 2.01 ± 0.86 [0.66–3.94] (n=20) | 2.21 ± 0.73 [1.18–3.79] (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 1.96 ± 0.80 [0.62–3.88] (n=20) | 2.29 ± 0.88 [0.76–4.33] (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 2.10 ± 0.97 [0.65–4.39] (n=20) | 2.28 ± 0.84 [1.21–4.19] (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 2.02 ± 0.97 [0.58–4.51] (n=20) | 2.30 ± 0.83 [1.11–4.45] (n=20) |
| All conditions | 2.02 ± 0.85 [0.58–4.51] (n=100) | 2.31 ± 0.80 [0.76–4.45] (n=100) |

### Dynamic re-routing time (per re-route) (s)

| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | – | 1.32 ± 0.07 [1.27–1.37] (n=2) |
| Morning peak begins during the trip (Tue 06:50) | 1.55 ± 0.85 [0.54–3.70] (n=15) | 1.46 ± 0.42 [0.69–2.06] (n=19) |
| Evening peak begins during the trip (Tue 16:50) | 1.74 ± 0.80 [0.81–3.79] (n=15) | 1.48 ± 0.36 [0.82–2.13] (n=19) |
| Road closure 2 min into the trip (Tue 08:10) | 3.08 ± 4.08 [0.55–15.16] (n=20) | 1.54 ± 0.43 [0.77–2.19] (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 1.77 ± 0.71 [0.60–3.39] (n=20) | 1.55 ± 0.53 [0.56–2.53] (n=20) |
| All conditions | 2.09 ± 2.32 [0.54–15.16] (n=70) | 1.51 ± 0.43 [0.56–2.53] (n=80) |

### Dynamic re-routing time (per trip) (s)

| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 0.00 ± 0.00 [0.00–0.00] (n=20) | 0.13 ± 0.41 [0.00–1.37] (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 1.16 ± 1.01 [0.00–3.70] (n=20) | 1.39 ± 0.52 [0.00–2.06] (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 1.31 ± 1.04 [0.00–3.79] (n=20) | 1.41 ± 0.48 [0.00–2.13] (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 3.08 ± 4.08 [0.55–15.16] (n=20) | 1.75 ± 0.89 [0.77–4.38] (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 1.77 ± 0.71 [0.60–3.39] (n=20) | 1.93 ± 1.10 [0.56–4.14] (n=20) |
| All conditions | 1.46 ± 2.17 [0.00–15.16] (n=100) | 1.32 ± 0.96 [0.00–4.38] (n=100) |

### Re-routes per trip

| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 0.00 ± 0.00 [0.00–0.00] (n=20) | 0.10 ± 0.31 [0.00–1.00] (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 0.75 ± 0.44 [0.00–1.00] (n=20) | 0.95 ± 0.22 [0.00–1.00] (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 0.75 ± 0.44 [0.00–1.00] (n=20) | 0.95 ± 0.22 [0.00–1.00] (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 1.00 ± 0.00 [1.00–1.00] (n=20) | 1.10 ± 0.31 [1.00–2.00] (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 1.00 ± 0.00 [1.00–1.00] (n=20) | 1.20 ± 0.41 [1.00–2.00] (n=20) |
| All conditions | 0.70 ± 0.46 [0.00–1.00] (n=100) | 0.86 ± 0.49 [0.00–2.00] (n=100) |

### Convergence rate: iterations to convergence (initial search) (iterations)

| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 25.70 ± 4.76 [21.00–37.00] (n=20) | 25.70 ± 3.06 [23.00–34.00] (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 24.70 ± 2.79 [21.00–29.00] (n=20) | 25.05 ± 1.32 [23.00–28.00] (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 25.55 ± 5.20 [21.00–39.00] (n=20) | 23.90 ± 1.55 [21.00–26.00] (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 25.50 ± 4.48 [21.00–35.00] (n=20) | 25.10 ± 1.97 [22.00–28.00] (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 25.20 ± 3.74 [21.00–33.00] (n=20) | 25.10 ± 1.97 [22.00–28.00] (n=20) |
| All conditions | 25.33 ± 4.20 [21.00–39.00] (n=100) | 24.97 ± 2.11 [21.00–34.00] (n=100) |

Runs that reached the iteration limit without converging (not included above): ANTROUTE 0, Cheng 0.

### Iterations to convergence (all searches) (iterations)

| Traffic condition | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Off-peak, no change (Tue 10:10, historical weather) | 25.70 ± 4.76 [21.00–37.00] (n=20) | 25.55 ± 3.18 [22.50–34.00] (n=20) |
| Morning peak begins during the trip (Tue 06:50) | 24.15 ± 2.88 [21.00–32.00] (n=20) | 23.18 ± 0.85 [22.00–25.00] (n=20) |
| Evening peak begins during the trip (Tue 16:50) | 24.68 ± 3.84 [21.00–36.00] (n=20) | 22.57 ± 1.09 [21.00–26.00] (n=20) |
| Road closure 2 min into the trip (Tue 08:10) | 25.20 ± 3.13 [21.00–31.50] (n=20) | 22.98 ± 0.95 [21.50–24.50] (n=20) |
| Clear to heavy rain 3 min into the trip (Tue 08:10) | 25.00 ± 2.88 [21.00–31.50] (n=20) | 22.92 ± 0.94 [21.50–24.50] (n=20) |
| All conditions | 24.95 ± 3.54 [21.00–37.00] (n=100) | 23.44 ± 1.95 [21.00–34.00] (n=100) |

## SOP 3. Significant difference between ANTROUTE and the baseline

Paired by trip and traffic condition (seeds averaged). d = Cheng − ANTROUTE; positive favours ANTROUTE for every measure. Two-sided Wilcoxon signed-rank test, W = min(W+, W−); the four p-values of each condition are Holm-adjusted. Significant when the Holm-adjusted p < α = 0.05.

### All conditions

| Measure | n | ANTROUTE mean | Cheng mean | Mean d | W+ | W− | W | p | Holm p | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
| Distance travelled | 50 | 9.06 | 12.34 | 3.28 | 1269.0 | 6.0 | 6.0 | < 0.0001 | < 0.0001 | **Yes** |
| Path cost | 50 | 35.41 | 62.05 | 26.64 | 1275.0 | 0.0 | 0.0 | < 0.0001 | < 0.0001 | **Yes** |
| Computational time | 50 | 3.48 | 3.63 | 0.15 | 962.0 | 313.0 | 313.0 | 0.0014 | 0.0028 | **Yes** |
| Convergence rate | 50 | 25.33 | 24.97 | -0.36 | 631.0 | 594.0 | 594.0 | 0.8537 | 0.8537 | No |

### Off-peak, no change (Tue 10:10, historical weather)

| Measure | n | ANTROUTE mean | Cheng mean | Mean d | W+ | W− | W | p | Holm p | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 8.89 | 11.12 | 2.23 | 54.0 | 1.0 | 1.0 | 0.0039 | 0.0117 | **Yes** |
| Path cost | 10 | 32.73 | 51.76 | 19.04 | 55.0 | 0.0 | 0.0 | 0.0020 | 0.0078 | **Yes** |
| Computational time | 10 | 2.01 | 2.58 | 0.57 | 51.0 | 4.0 | 4.0 | 0.0137 | 0.0273 | **Yes** |
| Convergence rate | 10 | 25.70 | 25.70 | 0.00 | 32.5 | 22.5 | 22.5 | 0.6445 | 0.6445 | No |

### Morning peak begins during the trip (Tue 06:50)

| Measure | n | ANTROUTE mean | Cheng mean | Mean d | W+ | W− | W | p | Holm p | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 8.86 | 11.03 | 2.17 | 55.0 | 0.0 | 0.0 | 0.0020 | 0.0078 | **Yes** |
| Path cost | 10 | 33.84 | 52.90 | 19.06 | 55.0 | 0.0 | 0.0 | 0.0020 | 0.0078 | **Yes** |
| Computational time | 10 | 3.17 | 3.60 | 0.43 | 40.0 | 15.0 | 15.0 | 0.2324 | 0.4648 | No |
| Convergence rate | 10 | 24.70 | 25.05 | 0.35 | 33.5 | 21.5 | 21.5 | 0.5801 | 0.5801 | No |

### Evening peak begins during the trip (Tue 16:50)

| Measure | n | ANTROUTE mean | Cheng mean | Mean d | W+ | W− | W | p | Holm p | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 8.97 | 11.66 | 2.70 | 52.0 | 3.0 | 3.0 | 0.0098 | 0.0293 | **Yes** |
| Path cost | 10 | 33.41 | 53.25 | 19.84 | 55.0 | 0.0 | 0.0 | 0.0020 | 0.0078 | **Yes** |
| Computational time | 10 | 3.26 | 3.70 | 0.43 | 41.0 | 14.0 | 14.0 | 0.1934 | 0.3867 | No |
| Convergence rate | 10 | 25.55 | 23.90 | -1.65 | 20.0 | 35.0 | 20.0 | 0.4707 | 0.4707 | No |

### Road closure 2 min into the trip (Tue 08:10)

| Measure | n | ANTROUTE mean | Cheng mean | Mean d | W+ | W− | W | p | Holm p | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 9.54 | 14.60 | 5.07 | 55.0 | 0.0 | 0.0 | 0.0020 | 0.0078 | **Yes** |
| Path cost | 10 | 39.01 | 77.17 | 38.15 | 55.0 | 0.0 | 0.0 | 0.0020 | 0.0078 | **Yes** |
| Computational time | 10 | 5.18 | 4.03 | -1.15 | 31.0 | 24.0 | 24.0 | 0.7695 | 1.0000 | No |
| Convergence rate | 10 | 25.50 | 25.10 | -0.40 | 25.5 | 29.5 | 25.5 | 0.8691 | 1.0000 | No |

### Clear to heavy rain 3 min into the trip (Tue 08:10)

| Measure | n | ANTROUTE mean | Cheng mean | Mean d | W+ | W− | W | p | Holm p | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
| Distance travelled | 10 | 9.03 | 13.27 | 4.24 | 55.0 | 0.0 | 0.0 | 0.0020 | 0.0078 | **Yes** |
| Path cost | 10 | 38.06 | 75.15 | 37.09 | 55.0 | 0.0 | 0.0 | 0.0020 | 0.0078 | **Yes** |
| Computational time | 10 | 3.79 | 4.24 | 0.45 | 42.0 | 13.0 | 13.0 | 0.1602 | 0.3203 | No |
| Convergence rate | 10 | 25.20 | 25.10 | -0.10 | 24.5 | 20.5 | 20.5 | 0.8398 | 0.8398 | No |

## Runs that failed

None.
