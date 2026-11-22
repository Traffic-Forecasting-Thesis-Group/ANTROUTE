# ANTROUTE vs Cheng (2023): frozen-manifest comparison

Generated 2026-11-22T09:52:18+00:00 · manifest 2026-11-09T03:57:48+00:00 · calibration f948d5a348dd · scipy 1.14.1

Live evaluation on saved HERE traffic, incident and weather captures of the test period. Roads without a live HERE speed use the median live congestion of matched roads of the same city and road type; mean live coverage of the evaluated captures: 19.6% of segments.

## Primary outcomes

Requests: 30; complete matched pairs: 30 (seeds [0, 1, 2, 3, 4], mean per request). d = Cheng − ANTROUTE; positive favours ANTROUTE. Two-sided Wilcoxon signed-rank, Holm over the four primary outcomes, α = 0.05.

| Outcome | n | ANTROUTE mean ± SD | Cheng mean ± SD | Mean d ± SD | Median d | W+ | W− | W | p | Holm p | Significant |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Common evaluation path cost | 30 | 42.9836 ± 15.6524 | 43.7323 ± 16.5364 | 0.7487 ± 3.9722 | 0.3401 | 283.0 | 182.0 | 182.0 | 0.3085 | 0.3085 | No |
| ACO search time (ms) | 30 | 5,612.1056 ± 1,116.7682 | 4,074.4049 ± 920.1051 | -1,537.7007 ± 425.5121 | -1,658.6880 | 0.0 | 465.0 | 0.0 | < 0.0001 | < 0.0001 | **Yes** |
| Distance travelled (m) | 30 | 12,220.2552 ± 4,403.2493 | 11,024.3094 ± 4,037.3460 | -1,195.9458 ± 1,054.7511 | -951.0085 | 10.0 | 455.0 | 10.0 | < 0.0001 | < 0.0001 | **Yes** |
| Convergence rate | 30 | 0.8573 ± 0.1018 | 0.8270 ± 0.1228 | -0.0303 ± 0.0731 | -0.0250 | 121.0 | 314.0 | 121.0 | 0.0369 | 0.0737 | No |

## All requests: outcomes

| Algorithm | Outcome | Runs |
|---|---|---|
| ANTROUTE (proposed) | converged | 107 |
| ANTROUTE (proposed) | iteration_limit | 43 |
| Cheng (2023) baseline | converged | 116 |
| Cheng (2023) baseline | iteration_limit | 34 |

## Secondary (descriptive)

| Measure | ANTROUTE mean ± SD [min–max] | Cheng mean ± SD [min–max] |
|---|---|---|
| Iterations to convergence (the limit for runs that did not converge) (iterations) | 51.44 ± 8.57 [30.00–60.00] (n=150) | 49.62 ± 9.30 [26.00–60.00] (n=150) |
| First iteration that found the final best route (iterations) | 34.34 ± 12.42 [10.00–60.00] (n=150) | 31.49 ± 12.05 [6.00–59.00] (n=150) |

Stop reasons of runs that returned a route: ANTROUTE (proposed): converged 107, iteration_limit 43; Cheng (2023) baseline: converged 116, iteration_limit 34
Share of runs that converged before the iteration limit: ANTROUTE (proposed): 71.3%; Cheng (2023) baseline: 77.3%
