# Routing contract, version 2

One definition of the ten routing criteria is shared by calibration, routing, dataset exports and explanations
(`src/routing/criteria.py`). The six-process architecture is unchanged: data preparation, route request and condition
snapshot, criterion weighting and cost, ACO optimization, dynamic route output and condition updates, and
performance evaluation.

## Three kinds of value

| Kind | Examples | Used for |
|---|---|---|
| Physical | `distance_m`, `travel_time_s`, `capacity_vph`, severities in [0, 1] | Simulation clock, display, exports |
| Criterion | the ten additive criteria below | CRITIC calibration, route totals, explanations |
| Scaled | criterion ÷ development maximum | Routing cost with fixed weights |

## The ten additive criteria (per directed segment)

L = length (km), T = traversal time (min), g = congestion severity, p = road-class severity, h = time-profile
severity, a = weather severity, K = directional capacity (veh/h), I and S = intersection and signal encounters at the
downstream node (one per traversed approach; geometry vertices are not nodes).

| Criterion | Formula | Unit |
|---|---|---|
| distance | L | km |
| congestion | L·g | km × severity |
| travel_time | T | min |
| intersections | I | encounters |
| signals | S | encounters |
| road_type | L·p | km × severity |
| time_profile | L·h | km × severity |
| weather | L·a | km × severity |
| flood_exposure | Σ portion_km × class severity | km × severity |
| capacity_burden | L / K | km per veh/h |

Every criterion is additive: splitting a road at a node that is neither an intersection nor a signal leaves each
criterion total, and the cost under a fixed calibration, unchanged. Capacity burden is a provisional study mapping, not
a universal published road penalty.

## Routing cost

```
scaled[e, j] = criterion[e, j] / development_max[j]
edge_cost[e] = Σ_j fixed_weight[j] × scaled[e, j]
route_cost   = Σ_e edge_cost[e]
```

Nothing is clipped. A value above its development maximum is flagged (`exceeds_development_max:<criterion>`), and
scaled values and costs may exceed 1. A positively weighted criterion needs a positive, finite maximum; a zero-weight
criterion contributes zero without dividing. Fixed weights and maxima never change during requests, closures or
replanning.

## Encodings

- **Flood exposure:** each line is intersected with the NOAH five-year hazard classes. Overlaps resolve to the highest
  class, lengths are WGS84 geodesic, and the result is `flood_km_class1..3` plus a coverage status (`mapped`, or
  `unknown` outside the dataset's extent). The class severity {1: 1/3, 2: 2/3, 3: 1} is an assumed scoring scenario,
  with the alternatives `equal` and `convex` declared for sensitivity. Mapped hazard is susceptibility, not current
  inundation or a closure.
- **Road class:** p_k = 1 − median free-flow speed of class k ÷ the largest class median. This is an assumed
  preference, not a published penalty.
- **Time profile:** h is the mean congestion of the road's group (city × road type) for its day category
  (weekday/weekend) and local hour, over development dates only. It falls back to road type, then to all roads. The
  clock hour itself is never a penalty.
- **Weather:** a is the assumed severity of the hourly category. Weatherstack `ws_precip_mm` is a 1-hour accumulation
  in mm (verified: the 24 hourly values sum to the daily total). Heavy rain is ≥ 7.5 mm/h and rain is > 0 mm/h. A
  description decides when precipitation is missing; when both are missing the observation is missing and the request
  fails. Weather is kept separate from flood exposure.
- **Capacity:** K = directional lanes × an assumed base saturation flow of 1,900 veh/h/lane × g/C (0.5 assumed at a
  signalised approach). Lanes are already per direction. Method and uncertainty are stored per segment.
- **Traffic:** the hourly table is an assumed congestion scenario: a seeded demand-to-capacity index passed through
  BPR. It is not observed or independently estimated demand. `bpr_time(T0, q, K)` exists for scenarios that supply
  independent demand q.

## Calibration (development data only)

`configs/historical_study.json` must declare `calibration_split` (development, tuning, test). Until it does,
`scripts.build_study_network` exits with `CalibrationBlocked` and `/study/status` reports the reason. No dates are
invented.

- Hourly CRITIC: for each development sample hour, the alternatives are all road segments and the criteria are the ten
  additive criteria, min–max normalized within the hour inside CRITIC. Constant criteria get weight 0 that hour and
  are excluded from every correlation and conflict sum.
- Fixed weight: fixed_weight[j] = mean of w[j, h] over the common valid-hour set H, with zeros kept. Undefined hours
  (all constant, or perfectly concordant columns) and ineligible hours (missing inputs) are excluded whole and logged.
- Development maxima are the largest criterion values over the development sample, kept separate from CRITIC's
  within-hour normalization.
- `data/processed/calibration.json` stores the hourly vectors, active masks, valid and excluded hours with reasons,
  weights, maxima, units, mappings, the Cheng references and input hashes. Its SHA-256 is the `calibration_id`.

## Search (shared kernel, `src/routing/aco_kernel.py`)

Both models use one kernel and differ only in their native edge cost:

- η = 1 / (native cost + d(target, destination) ÷ frozen distance reference + ε).
- The q0 exploitation branch picks the lowest edge ID on ties; otherwise the next edge is sampled.
- Pheromone has a colony stage and an iteration-best stage. Failed ants deposit nothing, and pheromone never falls
  below a positive floor.
- Stopping uses a cumulative window: B[k−p] − B[k] ≤ abs_tol + rel_tol·|B[k−p]|. The defaults (p = 20, tolerances 0)
  reproduce "no improvement for 20 consecutive iterations".
- q0 = 0, beta = 1 and an 18 s wall-time budget were frozen on the tuning period by `scripts/tune_search.py` (`outputs/search_tuning.json`). At the deadline the search returns its best route
  labelled `time_limit`; with no route it raises `SearchTimeLimit`. A disconnected graph raises `NoFeasibleRoute`.
- Cheng (2023) uses s_ij = 0.637·L/L_ref + 0.258·T/T_ref + 0.105·F/F_ref with fixed development references, so an
  unrelated closure does not change the remaining roads' scores. Flow is a simulation adaptation (inverse BPR of the
  assumed congestion). The printed eq. 9 runs only in the mode `cheng_reference_printed`.

## Dynamic replanning

At each node arrival, the simulator:

1. applies due events and forms a new immutable snapshot (Asia/Manila clock);
2. rescores the incumbent remainder (invalid if it uses a closed road);
3. runs a fresh-pheromone search when the trigger policy fires;
4. adopts the candidate only if the incumbent is invalid or the candidate is strictly better.

A tie, a worse candidate, a time limit or a failed search keeps a valid incumbent. Controlled experiments use the
common trigger policy `controlled_common`.

## Evaluation

`scripts/evaluate_sop.py` runs a frozen manifest (`configs/experiment_manifest.json`) of test-period requests.

- **Primary outcomes:** common evaluation path cost and ACO search time.
- **Unit:** one request, taking the median over a complete matched seed set. Execution order is counterbalanced.
- **Test:** two-sided Wilcoxon signed-rank with d = baseline − ANTROUTE and Holm adjustment over the two outcomes.
  Failures are reported and never counted as zero.
- **Secondary:** distance and convergence behaviour, reported descriptively. Dynamic trips are a separate,
  descriptive experiment.

These results are software verification and modeled-scenario evaluation. They are not external empirical
validation.
