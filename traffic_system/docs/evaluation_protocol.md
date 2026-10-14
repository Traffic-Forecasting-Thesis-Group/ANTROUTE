# ANTROUTE evaluation protocol

What the evaluation measures, how, and what has to be recorded with every result. The code is
the source of truth; each section names the file that implements it.

## 1. Scope: Metro Manila

Metro Manila is the scope of the title, the statement of the problem, and the routing.

- **Routing** runs on the whole-city road graph: 59,521 nodes, 147,197 directed edges
  (`app/risk_routing.py`, `build_full_graph`).
- **Congestion risk** is also given for every edge of that graph, in three tiers
  (`src/routing/dynamic_weight.py`, `TIERS`):

| Tier | Where | Where the risk comes from | Edges (v8 scores) |
|---|---|---|---|
| **Observed** | Camera subgraph (k = 8 hops around the CCTV intersections) | RADR STGNN → MLP decoder prediction, with camera observations among its inputs | 2,894 (2.0%) |
| **Propagated** | Within 1.5 km by road of an observed edge | Risk of the nearest observed edge, same window | 17,825 (12.1%) |
| **Assumed** | Farther from the camera subgraph | Estimated from the other inputs only: weather, flood hazard, reported incidents, clock and road class (`src/routing/context_risk.py`) | 126,478 (85.9%) |

The assumed-tier estimate is a ridge regression fitted on the decoder's own risk over the camera
subgraph, using the risk file's train-split windows. For the v8 scores it reproduces the decoder
with R² = 0.74 on the windows it was fitted on and **R² = 0.46 on held-out windows**. Most of
what it captures is clock, weekday and weather; its incident coefficient is about 0, because
incidents rarely fall on scored edges. `GET /health/model` reports the coefficients and R², and
each route reports its share of length per tier (`risk_sources`).

Limitation to state: the fit sees only the corridor around the cameras, so road types and flood
levels that are rare there are extrapolated, and nothing observes traffic on assumed-tier roads.

A route whose length is mostly in the assumed tier still shows a congestion level, read off the
estimate, with the caption **"Estimated from weather, road type and traffic reports"** (`congestion_note`). Only
if the estimate could not be fitted does the level fall back to "unknown".

### Check against Google Maps (`scripts/google_segment_reference.py`)

300 road segments (100 per tier), each 1–2 km and nearly straight on roads with free-flow speed
≥ 30 km/h, were queried at Mon 07:45 and Mon 17:45. Google's congestion index is its travel time
with predicted traffic divided by its time without traffic. Of 588 answered queries, 239 were
dropped because Google's route was more than 25% longer than the segment (detours or U-turns),
leaving 349. Results are in `outputs/google_eval/segments/`.

| Comparison of predicted risk with Google's congestion index | Spearman ρ | n |
|---|---|---|
| All segments | +0.12 (p = 0.02) | 349 |
| Observed tier | −0.45 (p < 0.001) | 116 |
| Propagated tier | −0.03 (n.s.) | 98 |
| Assumed tier | +0.38 (p < 0.001) | 135 |
| Within the AM slot only | −0.09 (n.s.) | 174 |
| Within the PM slot only | −0.22 (p = 0.003) | 175 |

The risk scores get time of day right: risk is higher at the PM peak, and so is Google's
congestion index (about 1.05–1.11 AM, 1.15–1.27 PM in every tier). They do not rank roads within
a time slot: the predicted risk varies by only 0.01–0.02 between segments, and on the camera
subgraph it ranks them in the opposite order to Google (ρ −0.42 AM, −0.61 PM). The assumed tier's
+0.38 comes entirely from the AM/PM difference. Caveat: Google gives typical predicted traffic
for an October Monday, not the recorded May days.

## 2. ETA is not evaluated

ETA is not a research question or a metric. No system's own travel-time estimate is scored, and
the ETA calibration is gone. The app still shows a duration, from `src/routing/eta_engine.py`,
always labelled **"estimated, not evaluated"** (`RouteOption.duration_note`).

## 3. Metrics (`src/routing/metrics.py`, `scripts/evaluate_routing.py`)

| Metric | Definition | Better | Tested |
|---|---|---|---|
| Route Optimality | Σ Google time of Google's own route ÷ Σ Google time of the system's route × 100 | higher | Wilcoxon signed-rank, paired by trip |
| Congestion exposure | Σ Risk(u, v) over the route's edges, under the window's risk | lower | Wilcoxon signed-rank, paired by trip |
| Event exposure | Σ Event(u, v) over the route's edges, under the incidents in force at the window | lower | Wilcoxon signed-rank, paired by trip |
| Avoidable event exposure | the same, leaving out edges that touch the trip's own stops | lower | Wilcoxon signed-rank, paired by trip |
| Route length | metres | n/a | descriptive only |

Δ% uses Equation 8 for Route Optimality and Equation 7 for the exposures. A positive Δ% always
means ANTROUTE did better. Event exposure includes the edges at a trip's own origin and
destination, which neither system can avoid; that unavoidable part is identical for both systems.
Avoidable event exposure leaves those edges out, so it measures only what routing could have
avoided. Both are reported.

## 4. Reference: Google Maps

Both systems' routes, and Google's own fastest route, are timed in Google Maps. Google only
predicts traffic for a present or future date, so each recorded window becomes a **departure slot**
(weekday and time, e.g. `Mon 17:10`). The lookup uses the next date that falls on that weekday.

Each lookup must record (`REFERENCE_FIELDS` in `src/routing/route_trials.py`):

| Column | What to record |
|---|---|
| `google_access_method` | e.g. `Google Maps web (desktop), Depart at`, or `Routes API computeRoutes, TRAFFIC_AWARE_OPTIMAL` |
| `google_departure_date` | the date entered as the departure (YYYY-MM-DD) |
| `google_queried_on` | the date the lookup was made (YYYY-MM-DD) |
| `google_traffic_model` | which figure was read. For the web app's "typically X–Y min" range, pick one rule (e.g. `typical_range_upper`) and use it for every lookup; for the API, its `trafficModel`. |

`scripts/evaluate_routing.py --google-maps` skips any trip whose rows leave one of these blank,
and writes a `reference` block to `metrics.json`: access methods, traffic models, departure-date
and query-date ranges, and departure slots. The app's Evaluation page shows this block.

Workflow:

```
python scripts/evaluate_routing.py --risk-edges data/processed/risk_scores/risk_edges.csv \
    --auto-labels <auto_labels.csv ...> --template outputs/appendix/template.csv
python scripts/citywide_trips.py --out outputs/appendix/template_citywide.csv
python scripts/google_maps_worklist.py --template outputs/appendix/template.csv      # one row per lookup, with a Google Maps link
#   fill duration_seconds + the four provenance columns, then:
python scripts/google_maps_worklist.py --template outputs/appendix/template.csv --apply
python scripts/evaluate_routing.py --risk-edges data/processed/risk_scores/risk_edges.csv \
    --google-maps outputs/appendix/template.csv outputs/appendix/template_citywide.csv
```

With the API, `scripts/google_routes_lookup.py --worklist <lookups.csv ...>` fills the worklist
instead of a person, recording the provenance columns itself (`--reuse` copies identical lookups
from earlier worklists, `--limit 1` for a trial request). The key is `GOOGLE_ROUTES_API_KEY` in
`traffic_system/.env`.

### Results (2026-10-14, `outputs/routing_eval/`, sheets in `outputs/appendix/`)

Reference: Google Routes API v2 `computeRoutes`, TRAFFIC_AWARE_OPTIMAL, traffic model BEST_GUESS,
system routes forced through six of their own points as via waypoints. Departures on Mon 19 Oct
2026 in 32 AM/PM slots matching the recorded windows; looked up 14 Oct 2026. Scores from
`stgnn_edge_v8.pt`. 145 paired trips: 88 among the camera intersections, 57 elsewhere in Metro
Manila (`citywide_trips.py`).

| Scenario | n | Route Optimality, ANTROUTE / baseline | Congestion exposure, ANTROUTE / baseline | Event exposure | Avoidable event exposure |
|---|---|---|---|---|---|
| All trips | 145 | 77.6% / 69.1% (+12.4%, p < 0.001) | 55.3 / 87.6 (−36.9%, p < 0.001) | 0.12 / 0.12 | 0.013 / 0.013 |
| Recommended | 59 | 83.0% / 72.7% (p < 0.001) | 41.5 / 72.4 (p < 0.001) | 0 / 0 | 0 / 0 |
| Multi-destination | 56 | 73.1% / 67.7% (p = 0.003) | 87.1 / 135.3 (p < 0.001) | 0 / 0 | 0 / 0 |
| Incident-exposed | 30 | 75.7% / 64.4% (p = 0.049) | 22.9 / 28.4 (p = 0.003) | 0.58 / 0.58 | 0.062 / 0.062 |

Route length (descriptive): ANTROUTE 10.0 km, baseline 14.5 km on average. Alternative routes,
which the baseline does not define, are shown in the app and not evaluated.

Read with these notes:

- Event exposure is identical for both systems: almost all of it lies on edges at a trip's own
  origin or destination, which no route can avoid. Avoidable event exposure is identical too
  (0.06 on the incident-exposed trips, both systems): where an avoidable incident remained on a
  route, both systems' routes met it equally. At routing time ANTROUTE avoided the mid-route
  incident on 28 of the 30 incident-exposed trips, but the baseline's route mostly missed it as
  well. So these trips do not separate the two systems on events; the Event-Aware layer's effect
  is shown by the condition-update test (section 6).
- 14 trials score above 100%: Google timed the system's forced route faster than the route it
  chose itself (TRAFFIC_AWARE_OPTIMAL does not always return its fastest). They are kept.
- Left out: 14 trips where ANTROUTE found no alternative route, and 5 where Google returned no
  route through one system's waypoints.

Apple Maps is no longer used: its sheets and results were removed, and these Google Maps results
are the ones the app's Evaluation page shows.

## 5. Event-Aware layer: part of the method (`src/routing/event_layer.py`)

The layer is what the "road-event conditions" of RQ2 rest on.

- **Cost term:** `W(u, v) = dist(u, v) · (1 + λ · Risk(u, v)) · (1 + μ · Event(u, v))`, with λ = 2.0.
- **Strength:** μ = 1.0 (`DEFAULT_MU`; app setting `EVENT_MU`, now 1.0 by default). An incident's
  base impact is 0.5 when one lane is occupied and 1.0 for two or more. It applies to every edge
  touching the incident's intersection. Overlapping incidents take the strongest, so Event stays in
  [0, 1]. At μ = 1 a fresh two-lane incident doubles the cost of those edges.
- **Decay:** linear from the base impact at the report time to 0 after 60 minutes:
  `Event = max over live incidents of base · (1 − age / 60 min)`.
- **Matching:** an MMDA alert counts only when its location names a landmark in
  `configs/event_landmarks.csv`; unmatched alerts are dropped.

For evaluation the layer is on. `evaluate_routing.py` refuses `--event-mu 0`, and `--no-events`
is the ablation. `citywide_trips.py` and `evaluate_condition_updates.py` refuse to run with
`EVENT_MU = 0`. `metrics.json` records all of the above under `antroute.event_layer`.

## 6. Condition updates (`src/routing/condition_update.py`, `scripts/evaluate_condition_updates.py`)

The harness plans a trip under conditions A, applies an update, recalculates, and compares, for
ANTROUTE and the baseline on the whole-city graph, routed by the app's own code and settings.

**Updates**

- `--update traffic`: a different recorded window. t1 is the first window of the same session at
  least 30 minutes after t0, bringing its predicted risk and the incidents in force then.
- `--update session`: a different recorded session. t1 is the same clock time in the nearest other
  recorded day's session (AM or PM).
- `--update event`: a road event. An MMDA incident (2 lanes, strength 1.0) is injected at a
  gazetteer intersection (`--intersection`, else one drawn per trip) and stays active for its
  60-minute decay; it is measured at `--event-age-minutes` (default 0). Risk is unchanged, so the
  incident is the only change. Origin and destination are drawn on opposite sides of it, and a
  trip is kept only when the old route runs into it.
- `--update recorded-event`: a real incident reported between t0 and t1 (the old
  `--require-events`).

**Measures**

- **Route Recalculation Response Time:** seconds per recalculation, from conditions B being loaded
  (the window's risk and the baseline's observations are cached first) to the new route, which
  covers applying the incidents plus the search. Median and 95th percentile per system, on the
  same machine in one run, plus a paired Wilcoxon test (ANTROUTE vs baseline).
- **Changes in Recommended Routes:** share of trips whose route changes, and the shared-length
  fraction, the share of the old route's length the new route still uses (1 = unchanged).
- **Route Quality Before and After:** the old route under conditions B vs the recalculated route,
  on congestion exposure and event exposure (avoidable event exposure alongside); route length is
  descriptive. Both systems are measured with the same yardstick: ANTROUTE's risk tiers and
  Event-Aware layer at B.
- **Warm start (optional, `--warm-start`):** ANTROUTE recalculates again starting from the old
  colony's pheromone (rescaled to the cold start's mean, with a floor), and reports convergence:
  the iteration at which the colony first reached its final best cost, cold vs warm.

```
python scripts/evaluate_condition_updates.py --update traffic --n-trips 30 --warm-start --out-dir outputs/condition_updates/traffic
python scripts/evaluate_condition_updates.py --update session --n-trips 30 --warm-start --out-dir outputs/condition_updates/session
python scripts/evaluate_condition_updates.py --update event   --n-trips 30 --warm-start --out-dir outputs/condition_updates/event
```

Report each update type as its own sample. `--google` adds Google Route Optimality for
ANTROUTE's old and recalculated routes; `--rescore` recomputes a finished run from its saved routes.

### Results (2026-10-15, `outputs/condition_updates/`, test split, 30 trips each)

Machine: Windows 11, Intel 12-thread CPU, Python 3.12; ANTROUTE colony 8 ants x 15 iterations;
baseline colony 8 x 15 in a corridor around the trip.

| Update | System | Response time, median / p95 | Routes changed | Shared length (changed routes, median) | Congestion exposure, old → recalculated | Event exposure, old → recalculated |
|---|---|---|---|---|---|---|
| Traffic (next window) | ANTROUTE | 0.34 s / 0.59 s | 2 of 30 (7%) | 0.93 | 73.2 → 73.0 (n.s.) | 0 → 0 |
| | Baseline | 0.24 s / 0.61 s | 22 of 30 (73%) | 0.40 | 115.7 → 123.4 (n.s.) | 0 → 0 |
| Session (other day) | ANTROUTE | 0.42 s / 0.60 s | 4 of 30 (13%) | 0.90 | 72.1 → 72.6 (n.s.) | 0 → 0 |
| | Baseline | 0.17 s / 1.06 s | 16 of 30 (53%) | 0.41 | 118.5 → 120.8 (n.s.) | 0 → 0 |
| Injected incident | ANTROUTE | 0.18 s / 0.24 s | 23 of 30 (77%) | 0.83 | 53.5 → 54.9 (n.s.) | 2.00 → 0.53 (p < 0.001) |
| | Baseline | 0.08 s / 0.40 s | 0 of 30 (0%) | 1.00 | 73.4 → 73.4 | 1.13 → 1.13 |

- Both systems recalculate a whole-city route in well under a second. The baseline is faster
  (paired p < 0.05 in all three samples), because it searches only a corridor around the trip.
- ANTROUTE keeps its route when only traffic changes (7-13% change, and changed routes keep about
  90% of their length). The baseline's routes swing far more (53-73% change, keeping about 40%),
  and those swings do not lower congestion exposure.
- When an incident lands on the route, ANTROUTE moves off it in 23 of 30 trips, cutting event
  exposure from 2.00 to 0.53 for a small, not significant rise in congestion exposure. The
  baseline has no incident input, so it never changes its route.
- Warm start brings no gain: cold and warm colonies both reach their final best at iteration 1
  and return the same route, because ANTROUTE's cost-to-go heuristic already steers the first
  ants onto the best route. The warm run's lower time follows from running second (warm
  caches), not from the pheromone.

The earlier samples in `outputs/google_eval/condition_updates/` (uniform: 0 of 30 changed; new
recorded incident: 15 of 30 changed, event exposure 0.70 → 0.36, Google Route Optimality 88.2% →
85.1%, n.s.) were ANTROUTE-only runs of the previous version of this script.

## 7. Model provenance

`risk_edges.csv` says nothing about the model that produced it. `predict_congestion_risk.py`
writes `risk_summary.json` beside it, and `evaluate_routing.py` now **requires** that file
(override: `--allow-unknown-model`) and copies it into `metrics.json` under `model`.

The current scores (Drive `MMDA_RISK_SCORES/`, written 2026-10-08):

| | |
|---|---|
| Checkpoint | `stgnn_edge_v8.pt` |
| Text branch (DistilBERT event text) | **on** (`use_text: true` in both the scoring summary and the checkpoint's own config) |
| Node context (weather, flood, incidents, clock, road) | on, nothing dropped |
| Decoder MAE (test split) | 0.318 over 1,123 labelled edges |

The `--no-text` note in `SESSION_HANDOFF.md` applies to `stgnn_edge_v1`/`v2` only. Of the later
checkpoints, only `stgnn_edge_v7` was trained without text.

### Does text actually move the scores?

`src/models/source_sensitivity.py` now has a `text` source (text silenced, as for a minute with
no tweets). `scripts/verify_crs_sources.py` on v8, 24 windows, with the embeddings from Drive
`MMDA_TEXT/embeddings.pt` (`outputs/google_eval/stgnn_edge_v8.sources_with_text.json`):

| Source removed or changed | Mean change in risk, camera edges | Mean change, other edges | Other edges that move |
|---|---|---|---|
| Text | 0.0000 | 0.0000 (max 0.008) | 0% |
| CCTV | 0.092 | 0.003 | 5.5% |
| Incidents (node context) | 0.008 | 0.009 | 83% |
| Weather | 0.067 | 0.067 | 100% |
| Flood | 0.062 | 0.077 | 100% |
| Time of day / weekday | 0.106 | 0.123 | 99% |

Tweets are present in only 0.1% of (camera, minute) slots, and removing them changes nothing.
Text is switched on in v8 but contributes nothing to the scores, so **the event-aware claim
cannot rest on the text branch.** It rests on the MMDA incident inputs in the node context (small
but network-wide) and on the Event-Aware layer (section 5), whose effect section 6 shows.

The local `data/processed/embeddings.pt` is stale (168 rows against 46,098 tweets); use the Drive
copy or regenerate it with `main.py`.

## 8. Input data: hourly weather and OSM road attributes

### Weather: hourly from WeatherStack (`src/ingestion/temporal_weather.py --hourly`)

`data/processed/temporal/weatherstack_hourly.csv`: WeatherStack historical API, `interval=1`,
for every day from 1 June 2025 to 31 May 2026 (the same span as the daily file and the tweet
collection) in each of the 34 weather grid cells (0.05°): 34 × 365 × 24 = 297,840 rows, in
Manila local time (UTC+8). Requested in runs of up to 60 consecutive days per cell; WeatherStack
counts each day toward the monthly quota (12,410 cell-days). Columns: temperature,
precipitation (mm per hour), humidity, plus wind, visibility, cloud cover and conditions for
later use. `python src/ingestion/temporal_weather.py --hourly` collects it and resumes where it
stopped; `--recorded-days-only` limits it to the camera-session days.

The old daily file (`weatherstack_historical.csv`, `interval=24`) does not describe the peak
hours the model sees:

| | Daily file | Actual AM peak (07–08) | Actual PM peak (17–18) |
|---|---|---|---|
| Temperature | ≈ 35 °C (the day's maximum) | 28.7–30.1 °C | 31.3–32.9 °C |
| Rain on 18 May | 7.2 mm (the day's total) | 0.0 mm/h | 0.49 mm/h |
| AM peak dry (no rain) | — | 9 of 10 days | — |

With daily weather both sessions of a day got the same values, so a dry morning on a rainy day
looked like rain.

Where it is used (each falls back to daily if the hourly file is absent):

- **CNN+LSTM input** (`alignment.build_sessions`): each minute of a session gets the city-wide
  weather of its own hour (`SessionRecord.weather` is then `[120, 3]`).
- **STGNN node context** (`node_context.NodeContext`): each road node gets its own grid cell's
  weather at that hour, and flood × rain uses that hour's rain.
- **Assumed-tier estimate** (`context_risk.EdgeContext`, refitted at every server start): now
  R² 0.53 on its fitting windows and **0.64 on held-out windows** (was 0.74 and 0.46 with daily
  weather).

Hourly rain is mm per hour and daily rain is the day's total, so a model must be scored with the
resolution it was trained on. Training records `weather_resolution` in the checkpoint;
`predict_congestion_risk.py` and `verify_crs_sources.py` pick the matching file by default and
refuse the other. Checkpoints without the key (v1–v9, including v8) are daily.

**v8 was trained on daily weather, so the current risk scores still use daily weather.** Hourly
weather reaches the risk scores only after retraining (`scripts/train_stgnn_edge.py`, which uses
the hourly file by default) and rescoring, then rerunning the evaluation.

### Road attributes from OSMnx (`src/ingestion/spatial_topology.py --edge-attributes-only`)

The stored graph (`data/raw/spatial/metro_manila_road_graph.graphml`, OSMnx 2.0.1, 59,521 nodes,
148,483 edges; 147,197 directed edges in the simple graph the model and router use):

| Attribute | Coverage | How missing values are filled |
|---|---|---|
| Length | 100% | — |
| Road type (`highway`) | 100% | — |
| One-way | 100% | — |
| Speed limit (`maxspeed`) | 22.7% posted | by road type: motorway 80, trunk 60, primary 50, secondary 40, tertiary 30, residential 20 km/h (`HWY_SPEEDS_KPH`) |
| Free-flow travel time | 100% | length ÷ speed |
| Lane count (`lanes`) | 51.5% tagged (99% on primary/secondary/trunk, 37% on residential) | by road type: motorway 3, trunk/primary/secondary 2, others 1 (`HWY_LANES_FALLBACK`) |
| Street name | 83.6% | — |

Built from the stored graph without re-downloading, so the node order and adjacency the
checkpoints are aligned to are unchanged (checked before writing):

- `metro_manila_speed_kph.npz`, `metro_manila_lanes.npz`
- `full_network_edge_features.csv`: per edge length, road type, lanes, speed, travel time,
  one-way, access, name, and `maxspeed_source` / `lanes_source` (`osm` or `road_type`):
  33,493 posted speed limits and 113,704 assumed; 76,065 tagged lane counts and 71,132 assumed.
- `metro_manila_travel_time.npz` (already present) matches the graph exactly.

Model inputs: until now the decoder's per-road inputs were length and free-flow speed only (speed
already carries the speed limit and travel time). New training adds **lane count and road class**
(`EDGE_FEATURES_V2`; road class is an ordinal from motorway 1.0 down to residential 0.15). Each
checkpoint records `edge_feature_names`, and one without it uses the original two, so v8 scores
exactly as before. The assumed-tier estimate also uses lane count and road class.

Access tags are kept as they are: 373 edges tagged `access=no` and 17,690 `access=destination`
(mostly gated subdivisions) stay routable for both systems, by the team's decision.

## 9. Making text reach the model

Section 7 found text contributes nothing to v8's scores because it reached only 0.1% of
camera-minutes. Measured on the 20 recorded sessions, the cause is mostly timing: of 46,098
tweets only 507 were posted during a session (or the hour before it), and a placed tweet counted
only in the single minute it was posted. Three changes, all used by new training:

| Change | Where | Setting |
|---|---|---|
| A tweet stays in the text input for 60 minutes after it is posted (the Event-Aware layer's incident lifetime); tweets from the hour before a session carry in | `alignment.build_sessions(text_active_minutes=)` | `--text-active-minutes 60` |
| A tweet reaches every camera within 2 km (straight line) of the place it names, not only that place's own camera | `tweet_geocode.cameras_within`, `load_tweets_table(radius_km=)` | `--text-radius-km 2` |
| Landmark list expanded: EDSA cross-streets Nominatim cannot place are located on the road graph where EDSA meets a road of that name (`source=road_graph`) | `scripts/expand_event_gazetteer.py` | 138 landmarks, 86 intersections (from 85 and 39) |

Result, through the real training pipeline on all 320 camera-sessions: **camera-minutes with
text 0.1% → 14.6%** (49 → 5,599 of 38,400); camera-sessions with any text 36 → 109. MMDA alerts
placed on the graph: 76% → 90% of 2,280.

Checkpoints record `text_radius_km` and `text_active_minutes`; scoring reads them, and a
checkpoint without them (v1–v9, including v8) is scored with the original rules (0 km, 1 minute).

The expanded landmark list is shared with the Event-Aware layer and the incident inputs of the
node context, so more incidents are now placed on the graph in routing too. The results in
sections 4–6 were computed with the old list; they change when the evaluation is rerun after
retraining. Whether text then measurably moves the scores is checked with
`scripts/verify_crs_sources.py`, whose `text` source now exists.

### The text prior: using the year of posts, not only the 507 during sessions

Live text is capped by the recordings: only 507 posts fall inside the 20 sessions (or the hour
before), and no placement rule can change that. The rest of the corpus describes the same roads
over the previous eleven months, so it enters as a **prior** (`src/data/text_prior.py`):

- **Placement on the road graph.** A landmark from the gazetteer places a post on the nodes
  within 300 m of that intersection; otherwise every major OSM road it names places it on that
  road's nodes (1,100 road names, plus aliases such as EDSA and C-5). #MakatiTraffic posts
  ("along Kalayaan Ave. from Makati Ave. to Rockwell Drive") are placed this way.
- **Features, per node and hour bucket** (24 hours × weekday/weekend): report rate, incident
  rate (crash, stalled, flood, …), and the mean DistilBERT embedding of those posts reduced to 8
  principal components (63% of the embedding variance). Ten new node-context columns,
  `text_reports`, `text_incidents`, `text_pc1`–`text_pc8`, appended after the original 17.
- **No leakage.** Only posts created before 1 May 2026 are used; every camera session (train,
  validation and test) is in May 2026. The checkpoint records the cutoff (`text_prior_before`).

| | Posts | Share of 46,098 |
|---|---|---|
| Placed on the road graph and before May 2026 (whole city) | 24,837 | **53.9%** |
| … of which concern the camera subgraph the STGNN runs on | 14,978 | 32.5% |
| Live, during a session or the hour before | 507 | 1.1% |

The prior gives 971 of the camera subgraph's 1,424 nodes, and 11,176 nodes city-wide, a text
history. The STGNN reads it on the camera subgraph; the assumed-tier estimate reads the report and
incident rates city-wide (its coefficients stay near 0 until a model trained with text exists to
imitate). Older checkpoints read only the first 17 columns and are unaffected. `--no-text-prior`
trains without it, and `verify_crs_sources.py` measures it as the `text_prior` source.

Placement takes about a minute and is cached in `data/processed/text_prior_placement.npz`
(gitignored, rebuilt automatically when the posts, gazetteer or road names change).

### Text in the assumed tier: learning from real congestion

On the 86% of roads no camera reaches, the other inputs have to carry the camera's share. That
only works if the assumed-tier estimate learns from real congestion: imitating the decoder's
risk (the original fit) gave text no weight, because the decoder never saw text. The estimate
(`app/risk_routing._context_model`) is now fitted on the **human congestion labels**
(`weak_target`, 6,594 labelled camera-edge rows from 209 train-split windows) and scored on the
2,830 rows of the validation and test windows. Its inputs: weather (hourly), flood, MMDA incident
impact, clock, road speed/length/lanes/class, the **text prior** (report and incident rates per
road and hour, posts before May 2026) and **live text** (placed posts of the last 60 minutes,
decaying like incidents; `text_prior.LiveText`, 27,117 placeable posts).

Held-out (`text_ablation` in `/health/model`):

| Inputs | MAE | Spearman |
|---|---|---|
| Constant at the training mean | 0.306 | — |
| No text | 0.273 | 0.245 |
| + text prior | 0.270 | 0.233 |
| + text prior + live text | **0.270** | 0.235 |

Text now carries weight comparable to the clock (standardised coefficients: live incident posts
+0.055, live posts −0.041, prior reports −0.036; time of day −0.045, incidents −0.042), and lowers
the held-out error by about 1%; it does not improve the ranking. The likely reason is the labels
themselves: they are relative to each camera (each camera's own terciles), so a signal that
mainly distinguishes places -- "this road is reported often" -- cannot predict them; only
signals that change over time (live posts) can.

Scale: the estimate ranks the far roads, and `context_risk.quantile_map` gives those ranks the
values of the decoder's risk on the same window's observed edges, so both tiers are priced on one
scale (Mon 17:40: observed mean 0.696, assumed 0.722, standard deviation ~0.03 each). The assumed
tier sits about one standard deviation higher, which slightly favours routes through the camera
area.

## Google's terms on storing results, and the Holm correction

- **Stored from Google:** only each route's duration in seconds, the departure slot and the query
  date (`outputs/appendix/lookups*.csv`), kept as evaluation evidence. Route geometry, place
  content and map tiles are not stored or republished. The Maps Platform terms limit caching and
  storing Google content, so confirm this use against the current Terms of Service before the
  lookup files go into a public repository or the paper's appendix. The files are committed
  nowhere until that is confirmed.
- **Holm correction:** the Q3 family, congestion exposure and event exposure, is corrected
  together (`significance.holm_adjust`, `Q3_FAMILY`); `wilcoxon_p_holm` and `significant_holm`
  are in `appendix3_significance.csv`, and the app's Evaluation tab shows the adjusted values.
  Route Optimality (Q2) and avoidable event exposure are each their own family of one.
- **Per-trip provenance:** `per_trip.csv` carries `observed_share_*`, `propagated_share_*` and
  `assumed_share_*` for each system: the share of route length in each risk tier (the assumed
  share covers what the pasted request called borrowed and default).

## 10. The split, said once

The v8 checkpoint (the one behind `risk_edges.csv`) was trained and scored on this split, read from
its own `config["visual_split"]` and the saved `configs/visual_split_v8.json`:

| Split | Sessions | Count |
|---|---|---|
| train | May 4 PM; May 6, 8, 11, 13, 18 AM and PM | 11 |
| validation | May 20 AM and PM; **May 22 AM and PM** | 4 |
| test (held out) | **May 25 AM and PM** | 2 |
| not used | May 4 AM; May 15 AM and PM (partial extraction, no usable labels) | 3 |

May 22 is labelled (about 1,000 human labels) and **validates**; it is not a test session and it is not
unlabelled. May 25 is the only held-out date: the model saw it in neither training nor validation.
Two older statements disagreed with this and were corrected: `SESSION_HANDOFF.md` called May 22
unlabelled, and `alignment.VISUAL_SPLIT` (the automatic 14 / 3 / 3 split over all 20 sessions, used only
when training is given no split file) puts May 22 PM in test. The risk file's `split` column is
the authority, and the app reads it.

## 11. The app's recorded inputs, simulation mode and what-if table

There is no live feed. Every input the app shows is **recorded**: read from one 30-minute window
of `risk_edges.csv` and the files beside it (`app/recorded.py`).

- **Inputs used panel** (`recorded_inputs` on `POST /routes/plan`, or `GET /routes/recorded-inputs`):
  the matched window with its split (and "held out" for May 25); the incidents in force then
  (location, kind, lanes, strength, minutes ago); that day's WeatherStack hourly weather (city mean
  at the window's hour); the Project NOAH flood hazard of the roads; and every camera with the
  model's risk on its edges, the human congestion label (Light / Medium / Heavy) and the mean YOLO
  vehicle count in the window (`scripts/build_vehicle_counts.py`).
- **Per-route fields** (`inputs` on each route option): the cameras and incidents within 200 m of the
  route, the weather across the grid cells it crosses, the flood level along it, the mean vehicle
  count at its cameras, and `risk_sources` (observed / propagated / assumed share of its length), so
  alternatives are compared on what each is made of.
- **Simulation mode** (`recorded_window` on the plan request; `GET /routes/recorded-sessions` lists
  the days and sessions): both systems plan on the chosen recorded window itself, with that day's
  overlays, whatever the departure time. May 25 is flagged held out. Simulated trips are not part
  of the Google Maps evaluation.
- **What-if table** (`GET /routes/what-if`, `scripts/build_what_if.py`): the real v8 checkpoint is
  re-run offline with one input changed at a time (heavy rain, flood level 3, new incidents, cameras
  off, no traffic posts, time of day) over 24 recorded windows, and the table shows how far the risk
  moves on camera roads and on the roads without a camera. It also shows what an injected incident does
  to a route, from the condition-update run (section 6). It is precomputed, not a live run: serving the
  checkpoint (PyTorch and the frames) on the app server is not done.
- **Hourly weather:** `weatherstack_hourly.csv` (June 2025 – May 2026) and `node_context.py` read it;
  the next retrain on Colab uses it. The retrained scores are not produced yet.
