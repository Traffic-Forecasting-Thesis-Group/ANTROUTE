# Non-visual Congestion Risk for roads outside the camera subgraph

This note records an architecture change to ANTROUTE's Congestion Risk Score (CRS) for the
thesis: what was wrong, what replaced it, how it was validated, and what it can and cannot
claim. Numbers come from `outputs/nonvisual_risk/report.json`, written by
`scripts/train_nonvisual_risk.py`.

## 1. The problem

The CRS is produced by CNN+LSTM → RADR-STGNN → MLP decoder (`src/models/traffic_risk_model_edge.py`)
and written to `risk_edges.csv` by `scripts/predict_congestion_risk.py`. That model runs only
on the k-hop **camera subgraph** (`build_subgraph`, k = 8): 2,894 of Metro Manila's 147,197 road
segments (2.0%). It cannot run on the whole city, because its GCN uses a dense normalised
adjacency matrix (about 14 GB for 59,521 nodes). Inside the subgraph, roads without a camera get
a learned placeholder plus non-visual context, and the model is trained only through the loss
on camera segments.

For routing, the app (`app/risk_routing.py` → `src/routing/dynamic_weight.fill_unscored_risk`)
filled the other 98% of segments as follows:

| Share of segments | Old rule |
|---|---|
| 2.0% | decoder prediction |
| 12.1% | borrowed from a predicted segment within 1.5 km by road |
| **85.9%** | **the window's median predicted risk: one constant for all of them** |

So the routing cost on most of the city was distance × a constant, which makes routing there
distance-only. None of these roads used their own weather, flood hazard, incidents, road
attributes or travel time, even though those data exist for the whole city.

**Could the existing model have been used with camera input set to zero?** No, for three
reasons:
- It never ran outside the subgraph, because of the dense adjacency.
- It was never trained with the visual modality absent at every node. Each training window has
  camera features at the 7 camera nodes, and those reach the rest of the subgraph through the
  GCN.
- It was never evaluated on any segment without a camera, since no labels exist there.

Passing zeros or the placeholder for an unseen road would be an unvalidated extrapolation.

## 2. The change

A separate **non-visual model** (`src/models/nonvisual_risk.py`) predicts risk from inputs every
road has, aligned to that road and that 30-minute window. It never uses camera features. The
absence of a camera is treated as "visual modality unavailable", not as "use a constant".

### Inputs

All inputs are built for every segment with sparse operations only (`RoadFeatures`).

| Group | Feature | Alignment |
|---|---|---|
| Weather | temperature, rainfall, humidity | the day, the segment's own 0.05° WeatherStack cell |
| Flood | flood hazard × that day's rainfall | Project NOAH 5-year level at the segment's endpoints |
| Incidents | live impact, live count | MMDA alerts within 0.5 km, decaying over 60 min, at the window's midpoint |
| Clock | minute of day, day of week (sin/cos) | the window |
| Availability | weather available; incident feed available; incident status covered | per segment and window |

The road-network and speed columns are also built for every segment, but the deployed model does
not use them (§4): length, free-flow speed and travel time, degree and speed at both endpoints,
flood level, and 2-hop neighbourhood means of speed, degree and flood. The 2-hop means are sparse
matrix–vector products over the 147,197-edge network. No dense city-wide matrix is built anywhere.

### Missing data is explicit

- **Incidents.** "No incident reported" (value 0, flags 1) is only claimed where an alert *would
  have been placed* had one been reported. That means the feed has data that day, and the segment
  is within 1.0 km of a point the gazetteer can locate an alert at (46 points, all on EDSA and
  Roxas Blvd). Everywhere else the incident value is **unavailable** (NaN, flag 0). On the
  routing graph, 8.7% of segments have a known incident status.
- **Weather.** A segment whose cell has no row for that day gets NaN with `weather_available = 0`.
  The current data cover the whole city for every day.
- **Learning the unavailable case (modality dropout).** The camera segments always have both
  sources, so training adds a copy of every row with incidents withheld and a copy with weather
  withheld. The gradient-boosted trees split on NaN natively, so the unavailable branch is learned
  rather than left arbitrary.

### How routing combines the layers

In `fill_unscored_risk`, each segment's source is recorded:

| Code | Source | Rule |
|---|---|---|
| 0 | `predicted` | the RADR-STGNN scored it (camera subgraph) |
| 1 | `borrowed` | within 1.5 km by road of a predicted segment: that segment's risk (**unchanged**) |
| 2 | `nonvisual` | the non-visual model's prediction for this window |
| 3 | `unavailable` | no prediction of any kind. Only when the non-visual model is not installed: routed on length alone (risk 0), never the median |

The window median is no longer used anywhere.

### Output metadata

- `GET /health/model` reports, for the serving window, how many segments came from each source,
  and which non-visual model is loaded (training split, rows, features).
- Every route option (`POST /routes/plan`) carries `risk_sources`: the share of the route's
  length that is predicted, borrowed, non-visual or unavailable. `risk_coverage` is now the share
  that is *not* unavailable.

Live, for the 2026-05-22 08:30 window:

| Source | Segments | Share |
|---|---|---|
| predicted | 2,894 | 2.0% |
| borrowed | 17,825 | 12.1% |
| nonvisual | 126,478 | 85.9% |
| unavailable | 0 | 0.0% |

## 3. Training and validation

**Labels.** `weak_target` (Light/Medium/Heavy as 0/0.5/1) exists only on the **42 camera road
segments at 9 camera intersections** (EDSA and Roxas Blvd): 9,424 rows over 9 days. The split is
the CRS's own whole-date train / val / test split.

**Model.** `HistGradientBoostingRegressor`: depth 2, 150 iterations, at least 80 rows per leaf,
L2 penalty 2. It is deliberately shallow because there are only 9 labelled places.

**Baselines.** All are evaluated on the same held-out rows:
- training-label mean;
- training mean per session (AM/PM peak);
- the window median fallback being replaced.

Two references are not available for an unseen road:
- the camera model's own prediction at the segment;
- the segment's own training mean per session.

MAE and RMSE are on the 0–1 scale; lower is better.

| Evaluation | Non-visual | Train mean | Session mean | Median fallback | Camera model (ref.) | Segment history (ref.) |
|---|---|---|---|---|---|---|
| Val (fit on train) | **0.251** / 0.311 | 0.293 / 0.316 | 0.284 / 0.309 | 0.264 / 0.312 | 0.255 / 0.301 | 0.249 / 0.290 |
| Test (fit on train+val) | **0.315** / 0.359 | 0.322 / 0.360 | 0.318 / 0.358 | 0.332 / 0.364 | 0.318 / 0.357 | 0.279 / 0.334 |
| **Leave one camera out, test days** | **0.323** / 0.367 | 0.324 / 0.362 | 0.322 / 0.362 | 0.332 / 0.364 | — | — |

**Leave one camera out** is the setting that matters. It trains on the other cameras' train+val
days and predicts the held-out camera's test days. That is the closest available stand-in for a
road the model has never seen labelled, which is how it is used. There:
- it is **no worse than a constant**: MAE 0.323 vs 0.324 for the training mean and 0.322 for the
  session mean;
- it is slightly better than the median fallback it replaces (0.332), in MAE but not in RMSE;
- its predictions vary more by place and time (SD 0.051 vs 0.004 for the median);
- it is **not shown to be more accurate than a constant**, so more varied predictions must not be
  read as more accurate ones.

**Response checks.** These measure response, not accuracy: the mean change in predicted risk on
3,000 random segments at 17:30 on 2026-05-25.

| Change | Effect |
|---|---|
| +20 mm rain | +0.013 |
| A live incident (impact 1) | +0.035 |
| AM-peak clock instead of PM | −0.011 |
| Halving free-flow speed, or maximum flood level alone | 0.000 (not model inputs; flood acts through flood × rain) |

## 4. Why the road-network and speed features are not used

They were requested, and they are built. With every built column, leave-one-camera-out MAE
**rose to 0.336**, worse than a constant (0.324). Exploratory variants gave 0.335–0.377 whenever
road class, speed or degree was included. With 9 labelled places, all on major arterials, those
columns identify *which camera* a row comes from, which helps on later days at the same cameras
but hurts on an unseen road. They should be revisited once labels exist on varied road types:
side streets, other cities, other road classes.

## 5. Limitations

1. **No ground truth off the cameras.** Every number above is measured at 9 camera intersections
   on two arterials. Nothing checks the non-visual predictions on the 126,478 segments they are
   used on.
2. **No better than a constant on unseen places.** The model's contribution is that the risk
   follows each road's own weather cell, flood exposure × rain, nearby incidents and the clock.
   That is not a demonstrated accuracy gain. On the held-out test rows, permutation importance
   is measurable only for time of day: about 0.002 MAE for `minute_cos`, 0.0005 for `minute_sin`.
   Weather and incidents add nothing measurable there. The model does respond to them (§3), but
   9 days of daily weather and few live incidents at the cameras cannot show that the response
   is right.
3. **Incident coverage is narrow.** Incident status is known on 8.7% of segments, those near
   EDSA and Roxas Blvd. Elsewhere it is unavailable, and the model falls back to the learned
   unavailable branch.
4. **Weather is daily.** WeatherStack history is per day and per 0.05° cell, so it cannot
   separate a morning storm from an evening one.
5. **Scale mismatch between layers.** On 2026-05-25 at 17:30 the non-visual layer averages 0.771
   (SD 0.049), while camera-model predictions average 0.696 (SD 0.025) and borrowed values 0.677.
   The non-visual model is fitted directly to the label scale, while the camera model's outputs
   are compressed and run low. With λ = 2 this makes segments outside the camera area about 7%
   more expensive for a reason unrelated to congestion, which pulls routes toward the camera
   corridor. Both models have since been calibrated onto the label scale: the camera model on
   the train split, the non-visual model out of fold (`docs/training_v9.md`). That narrows the
   gap by about 15–25% (17:30: +0.092 → +0.072) but does not close it, because the remainder
   comes from the non-visual model meeting roads unlike the cameras. The app routes on raw scores
   unless `RISK_SCORES=calibrated` is set.
6. **The routing evaluation predates this change.** The Apple Maps evaluation (50 trips) was run
   with the median fallback. The 30 camera-area trips are almost unaffected, since their routes
   are 100% predicted. The 20 city-wide trips were routed under the old fill, and their ANTROUTE
   routes may now differ.

## 6. Reproduce

```
# train, validate, and install the model (also places the MMDA alert feed once, ~3 min)
python scripts/train_nonvisual_risk.py

# tests: median removed, borrowing kept, explicit missing inputs, response to inputs
python -m pytest tests/test_dynamic_weight.py tests/test_nonvisual_risk.py

# the app loads data/processed/risk_scores/nonvisual_risk.joblib at the first route request
docker restart traffic-system
curl localhost:8000/health/model   # risk_sources, nonvisual_model
```

No city-wide risk file is written. The app predicts the non-visual layer per window on demand
(about 147k segments per window, cached for 16 windows).
