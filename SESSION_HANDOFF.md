# ANTROUTE — Session Handoff (as of 2026-10-09)

Read this first in a new session before doing anything else. It says where the project actually
stands against the system architecture, what's already working, what's broken, and what's next.

Branch: `fix/clean-code-structure`. Latest commit at time of writing: `726eb298`.

**This doc was last written at `40b1074d` (2026-09-30). 64 commits landed between then and now.** The
biggest change: the routing half of the thesis — Dynamic Weight Engine, Dynamic Routing Engine,
Optimal Path — went from "not started" to built, tested, and fully evaluated with significance tests.
There is now also a FastAPI backend and a working Expo app on top of it.

## Architecture status, stage by stage

| Diagram stage | Status |
|---|---|
| Unstructured Data (X/Twitter) → ... → DistilBERT | Done, and **now actually used in training** (the old doc's `--no-text` is no longer the operating mode). `data/processed/embeddings.pt` (768-d DistilBERT, 46,098 tweets). Tweets are placed per-camera via gazetteer match — a tweet naming no monitored intersection is dropped, because broadcasting every tweet to every camera made the text feature identical across cameras and therefore useless. **News/GDELT was removed from the project** — `gdelt_scraper.py` / `news_scraper.py` no longer exist. `pipeline_context.md` still documents them and is stale on that point. |
| Visual Data (MMDA CCTV) → Frame Extraction & Normalization | Done. Frames extracted as plain JPEGs (`scripts/extract_frames.py`), 1/min, across 10 session dates (May 4–25, AM+PM). A handful of chunks — mostly the Ayala `8337` camera — still fail on every method tried; low priority. |
| 2D Patch Embedding | First learned layer inside `CNNLSTMFusion` (`src/vision/patch_embedder.py`), trained end-to-end rather than a random fixed projection. |
| CNN + LSTM (Visual and Event Feature Extraction) | Implemented, `src/models/cnn_lstm_fusion.py`. |
| Temporal Data (WeatherStack) → Sequence Norm | Done, and now part of per-node context rather than a single global feature. |
| Spatial Data (OSMnx, Project NOAH) → Geocoding & Topology | Done. `data/processed/spatial/` has the full 59,521-node Metro Manila graph. **Flood hazard is now wired in** (`data/raw/spatial/MetroManila_Flood_5year.shp`, Project NOAH). Training uses a k=8 subgraph around the camera intersections (~1,409 nodes); **routing uses the full city graph**, not the subgraph. |
| RADR STGNN (GCN + GRU) | Implemented, `src/models/radr_stgnn.py`. Now fed a **17-feature per-node context vector** (`src/data/node_context.py`) in 5 ablatable groups — weather, flood, events, temporal, spatial — so non-camera nodes are no longer a single shared placeholder. `context=True` is the default; `--no-context` is the CCTV-only ablation. |
| MLP Decoder → Congestion Risk Score (per edge) | Done. `traffic_risk_model_edge.py` / `train_stgnn_edge.py` / `predict_congestion_risk.py` is **the pipeline**. Scores are now **calibrated** (`src/models/risk_calibration.py`) and shipped as a second column alongside the raw score. The old per-node pipeline (`traffic_risk_model.py` / `train_stgnn.py` / `predict_risk.py`) is still present and still tested — see "Open decisions". |
| Dynamic Weight Engine, Dynamic Routing Engine, Optimal Path | **Built and evaluated.** `W = d·(1 + λ·Risk)` with `λ = 2.0` in `src/routing/dynamic_weight.py`; ACO with a Dijkstra-derived goal heuristic in `src/routing/aco_routing.py`; Cheng (2023) Improved ACO baseline in `src/routing/baseline_iaco.py`. 50-trip evaluation against Apple Maps with Wilcoxon tests in `outputs/routing_eval/`. |

## Model status: v6 is what's live, v7 is trained but unscored

**No `.pt` checkpoint is in the repo** — they live on Drive in `MMDA_CHECKPOINTS/`. v1–v5 leave no trace
in the tree; only v6 and v7 are referenced anywhere.

- **v6 is canonical right now**: it's the checkpoint behind the only scored artefact that exists,
  `data/processed/risk_scores/risk_edges.csv` (120 MB, 2026-10-07), and it's what the API serves.
  Provenance is recorded in `risk_summary.json` and answerable live at `GET /health/model`.
- **v6 is the CCTV-only model** — i.e. it predates the multimodal context work. **90% of its scored
  edges carry the same risk in every one of the 366 windows.** So none of the "multimodal across the
  whole graph" work is reflected in any number the app currently reports.
- **v7 exists only as prose and tests.** No v7 scores, metrics file, or audit output is committed.

### Real v6 numbers (verified, from `risk_calibration.json` — MAE / RMSE / R², fit on train only)

| Predictor | train | val | **test** |
|---|---|---|---|
| ANTROUTE raw | 0.3279 / 0.3716 / −0.119 | 0.3159 / 0.3591 / −0.244 | **0.3626 / 0.3990 / −0.418** |
| ANTROUTE calibrated | 0.2929 / 0.3397 / +0.065 | 0.2883 / 0.3207 / +0.008 | **0.3049 / 0.3389 / −0.023** |
| always 0.5 | 0.3412 / 0.4130 / −0.383 | 0.3419 / 0.4134 / −0.649 | **0.3862 / 0.4395 / −0.721** |
| always 0.717 (train mean) | 0.3158 / 0.3513 / 0.000 | 0.2979 / 0.3247 / −0.017 | **0.3121 / 0.3417 / −0.040** |

**Read this honestly: raw v6 loses to a constant on every split.** Calibrated v6 beats the train-mean
constant on test by ~2.3% MAE (0.3049 vs 0.3121), with R² still slightly negative. Class separation is
real but thin — mean prediction on test is Light 0.516 / Medium 0.566 / Heavy 0.577, correct ordering,
~0.06 of separation. Pearson r on test is 0.156.

Two things not to do: **don't quote the `infer` split MAE (0.3096)** — it covers 8 labelled rows and
`mae_reliable` / `MIN_LABELLED_FOR_MAE = 50` exist specifically to flag it as not a held-out result.
And **don't cite the notebook's stored outputs** — they're from an older `stgnn_edge_thesis` run
(18 sessions, 11/3/3 split) and are not v6 or v7.

**There is no accuracy / F1 / AUC anywhere.** Risk is scored purely as regression against the
0.0/0.5/1.0 weak target. Don't let the paper claim a classification metric.

### What the v7 audit (`246fc067`) found

1. **Class weights were leaking** — they were counted over the whole label set, letting the held-out
   mix shape the loss. `train_split_lookup()` now counts training sessions only.
2. **Balanced class weights build the miscalibration in by construction**: they make the loss's best
   constant exactly 0.5 while the labels average ~0.72. `train_stgnn_edge.py` now prints a runtime
   NOTE recommending the **unweighted** loss for calibrated risk. v6's calibration exists to undo
   precisely this offset.
3. **CCTV was nearly inert in v7** — the sensitivity check found cameras moving camera-edge risk by
   only ~0.002 on average, because at `lr_graph = 10 × lr_fusion` the clock/weather context fits the
   labels long before the CNN+LSTM learns anything from frames. `--lr-context` was added as the fix;
   **the `--lr-context 1e-4` run has not been done.**
4. **The model does not clearly beat its baselines.** `src/models/risk_baselines.py` scores it against
   five train-fitted references, including an `edge × AM/PM history` forecaster using only road
   identity and time of day. **That strongest baseline has never been run on real data** — no output
   file exists. Whether the model beats it is still an open question.

### Calibration (`468f03d0`)

A single affine map, `risk_calibrated = clip(0.6901 × risk + 0.3225, 0, 1)`, least-squares fitted on
the train split's 6,594 labelled camera edges and applied unchanged to val/test. Affine rather than
isotonic/Platt because there are only 3 target values over ~6.6k rows. It is monotone, so **every edge
ranking and therefore every route is unchanged by construction** — it buys calibration, not routing
quality. Both columns are kept so raw and calibrated can be reported side by side.

## Routing results (verified, `outputs/routing_eval/`, n = 50 trips)

Run config: ANTROUTE λ=2.0, γ=1.0, event layer on (μ=1.0); baseline Cheng IACO α=1.0, β=5.0, ρ=0.3,
q0=0.7, 20 ants × 60 iterations. Ground truth is Apple Maps "typical traffic" for both systems.

| Metric | ANTROUTE | Baseline IACO | Δ% | Wilcoxon p | Verdict |
|---|---|---|---|---|---|
| **Route Optimality** | **83.66% (SD 13.86)** | 73.60% (SD 13.34) | **+13.66%** | **4.4e-6** | **ANTROUTE wins** |
| MAE (s) | 1095.0 | 826.2 | −32.5% | 0.0031 | baseline wins |
| RMSE (s) | 1121 | 851 | −31.7% | 0.0028 | baseline wins |
| MAPE | 51.8% | 37.8% | −37.0% | 5.3e-9 | baseline wins |
| R² (n=18) | −57.5 | −141.1 | +59.2% | 0.0139 | ANTROUTE wins |

**The finding is split, and that's a real result, not a bug: ANTROUTE wins decisively on route
quality and loses on ETA accuracy.** The baseline estimates ETAs from observed travel times; ANTROUTE
derives them from predicted risk via `t = t_freeflow · (1 + γ·Risk)`, and that's where it gives ground.
Report both.

Per-scenario breakdown exists for `recommended` (20), `alternative`, `multi_destination` (20), and
`incident_exposed` (10) — e.g. ANTROUTE 87.74% optimality on `incident_exposed`, 81.32% on
`recommended`. Note `recommended,route_optimality` alone is **not** significant (p = 0.064).

### How risk gets into routing

`predict_congestion_risk.py` → `risk_edges.csv` → `dynamic_weight.load_risk_edges` → joined by
**node-pair, not row order** → ACO. The decoder scores only **42 of 2,894 edges (~1.5%)**, so HEAD's
commit added a fill: an unscored edge within **`MAX_FILL_M = 1500` m** by road borrows the nearest
scored road's risk, else takes the window median. This fixed a real bug — the old flat
`DEFAULT_MISSING_RISK = 0.0` made unscored roads look congestion-free and the router "steered off EDSA
onto side streets". Each route now reports `risk_coverage` (share of its length that is
model-informed), and `congestion_level` is reported as **`"unknown"`** below
`MIN_RISK_COVERAGE = 0.5` rather than a false "clear". **State routing claims with this coverage
caveat** — the code already does.

## Application layer

**Backend** — FastAPI, `traffic_system/app/` (~2,150 lines). It loads **no `.pt` at serve time**; it
reads the pre-scored `risk_edges.csv`. Startup warms the road network on a daemon thread (~40 s).
Endpoints: `POST /routes/plan` (ANTROUTE vs baseline, departure time), `GET /routes/comparison-metrics`,
`POST /routes/trip-evaluation`, `GET /routes/forecast-metrics`, `GET /health/model`, plus `/auth/*`
(JWT) and `/places/*`. Geocoding is **OpenStreetMap Nominatim, not Google** — `GOOGLE_PLACES_API_KEY`
in config is dead. In the live app ACO runs light (8 ants × 15 iterations: identical path to 20×60 on
a 165-hop cross-city route, 0.9 s vs 8.7 s).

**Frontend** — React Native + Expo (SDK ~57, TypeScript), `frontend/`. `HomeScreen.tsx` is the whole
product: map with ANTRoute/baseline polylines, origin by search or GPS, multiple reorderable
destinations, departure-time picker, up to 3 route options with congestion/duration/distance/via,
a model toggle, a Comparison Result card and an Evaluation Metrics table fed by the real
Apple-Maps-scored evaluation, and a draggable bottom sheet. `FeedScreen` is **static mock data** and
makes no backend call — the only unimplemented user-facing surface.

## How to run it (the README is wrong here)

`README.md` has **two empty fenced code blocks** where the backend command should be (around lines
124–128 and 151–155), and one of them refers to a `backend` directory that doesn't exist. `uvicorn`
appears in **no** Markdown file in the repo. The real command, from `docker-compose.yml` and the
Dockerfile:

```bash
# from traffic_system/ — CWD matters, data/ and outputs/ resolve relative to it
uvicorn app.main:app --reload
```

**The `.venv` cannot run the backend.** It has torch, networkx, pydantic and uvicorn but **no
`fastapi`, `sqlalchemy`, `redis`, `osmnx`, `geopandas` or `passlib`**. This also means the old doc's
test command now hard-stops during collection:

```
ERROR collecting tests/test_comparison_metrics.py
E   ModuleNotFoundError: No module named 'fastapi'
323 tests collected, 1 error
```

So either `pip install -r traffic_system/requirements.txt` or use Docker. Installed `pydantic` (2.13.5)
and `pytest` (9.1.1) have also drifted from the pinned 2.9.2 / 7.4.0.

Also: `docker-compose.override.yml` remaps the backend to host port **8100**, while the frontend's
fallback is **8000** — and `frontend/.env.local` pins `EXPO_PUBLIC_API_URL` to a **dead trycloudflare
tunnel**. Fix that before trying to run the app.

**There is no CI.** `.github/` contains only `pull_request_template.md` — no `workflows/` at all,
despite `README.md:44` claiming "CI status checks must pass before a merge is allowed."

## Test suite

26 test files + `conftest.py`, 4,819 lines, **323 tests collected**. The newer routing/risk code is
well covered: `test_dynamic_weight.py` (388), `test_aco_routing.py` (269), `test_baseline_iaco.py`
(231), `test_node_context.py` (319), `test_routing_metrics.py` (190), `test_risk_calibration.py`,
`test_risk_baselines.py`, `test_event_layer.py`, `test_risk_routing.py`, `test_comparison_metrics.py`
(drives the API through `TestClient`), `test_eta_engine.py`, `test_departure_window.py`,
`test_day_split.py`. Nothing in the repo records a pass/fail run, so the old doc's claim that Windows
multiprocessing flakiness is "known and harmless" is unverified — but it's not the current blocker;
the missing deps are.

## Data / labelling status

- **20 sessions over 10 dates** (May 4, 6, 8, 11, 13, 15, 18, 20, 22, 25 2026 × AM/PM). The old doc's
  15 sessions and "May 22 unlabeled" are both obsolete — **May 15 and both May 22 sessions are now in
  the split.**
- **Official split is 14 / 3 / 3 sessions**, not 11/2/2 (`src/data/alignment.py:44`):
  ```
  train (14): May 04, 06, 08, 11, 13, 15, 18  (AM+PM each)
  val   (3):  May 20 AM, May 20 PM, May 22 AM
  test  (3):  May 22 PM, May 25 AM, May 25 PM
  ```
  Note **May 22 straddles val/test** — which is exactly what `--split auto-day` and
  `split_days_overlap()` were added to catch. `train_stgnn_edge.py` now warns if a date is in two
  splits.
- Verifiable label counts: **9,424 labelled camera-edge rows** in the v6 artefact (train 6,594 / val
  1,151 / test 1,679 / infer 8), class mix **58.5% Heavy / 30.1% Medium / 11.5% Light**. Frame-level
  label totals live on Drive with the frames and are **not** determinable from the repo — treat the
  old 12,178 figure and the notebook's 12,755 as unverified.
- Tweet gold set: `data/labels/tweet_training_window.csv` — 97 rows, 13 relevant / 84 not.
  **Honest finding from `19035c81`: those 97 labels add nothing on this corpus.** All 13 relevant
  incidents are at Pasay, Tandang Sora, Pasong Putik, Pasig, Caloocan and Taguig — none at a monitored
  junction. In-session placements are 69 with or without them. Over 40 hours of CCTV, nothing reported
  on X happened where the cameras were looking. What *did* move the number was the camera remap:
  29 → 69 in-session placements.

## Camera → node mapping: the old limitation is closed

**16 cameras now map to 9 distinct intersections** (`configs/camera_nodes.csv`), and **Ortigas and
Shaw are separate graph nodes** (`21834288` and `8462613410`) — `full_network_static_features.csv` has
exactly 9 `is_cctv_node` nodes out of 59,521. Two cameras had been genuinely mis-sited: **EDSA-Aurora
sat 10.8 km away** at EDSA × Taft in Pasay, and the merged EDSA-Ortigas-Shaw sat on Ortigas Ave while
the camera looks at Shaw Blvd, 1.4 km off. Both were silently corrupting training targets and every
risk score derived from them. The Ayala `8337` camera had been absent from the mapping entirely and
its frames were being dropped; it's now mapped.

Two residual inconsistencies: `configs/cctv_locations.csv` **still carries the merged
`EDSA-Ortigas-Shaw` label** and is stale relative to `camera_nodes.csv`; and it lists 2 cameras at
Roxas-Padre Burgos against 3 in `camera_nodes.csv`.

**Important:** after any siting change the k-hop subgraph changes shape, so **`risk_edges.csv` must be
regenerated** before its numbers mean anything. Source of truth is `is_cctv_node` in
`full_network_static_features.csv`, not the derived `key_intersections_node_order.csv`.

## Known limitations (state these plainly in the paper, don't hide them)

- **The decoder scores ~1.5% of city edges** (42 of 2,894). Everything else is borrowed within 1.5 km
  or median-filled. `risk_coverage` quantifies this per route; routing claims need the caveat.
- **The served model is CCTV-only v6** — 90% of its edges are constant across windows, so the
  multimodal work isn't in any reported number yet.
- **Raw v6 loses to a constant baseline**; only the calibrated score edges past it, by ~2.3% MAE.
- **No true per-timestamp sync across modalities.** Weather joins by date; tweets are placed by
  gazetteer match, not timestamp-aligned per camera. Explicit, discussed tradeoff.
- **Congestion labels are per-camera-relative** — "Heavy" on one camera's tercile split isn't the same
  real-world level as "Heavy" on another.
- **Labels are heavily skewed** (58.5% Heavy), which is what makes constant baselines competitive and
  why the weighted-vs-unweighted loss question matters.
- **ANTROUTE's ETAs are worse than the baseline's** on every error metric.
- **The event layer's status is contradictory.** `app/config.py` ships `event_mu = 0.0` (off), but
  `evaluate_routing.py` defaults it **on** at μ=1.0 and `metrics.json` records "event_layer: on". So
  the published 83.66% includes a component whose own module header says it is **"NOT PART OF THE
  THESIS METHOD"** and must be documented before any result using it is reported. **Resolve this
  before writing results.**
- Security, if this is ever exposed: `SECRET_KEY` defaults to `"change-me-before-you-ship-anything"`,
  CORS is `*`, migrations are `create_all` at startup (Alembic installed but unused), and Nominatim
  still carries a placeholder contact e-mail in its User-Agent.

## Immediate next steps

1. **Rescore with a context (multimodal) checkpoint.** This is the blocking step for almost everything
   else — every number the app reports is currently from CCTV-only v6.
2. **Run the risk baselines for real.** `evaluate_risk_baselines.py`, `verify_crs_sources.py` and
   `source_sensitivity.py` all exist and **none has ever written an output**. Specifically, the
   `edge × AM/PM history` baseline is the one that decides whether the model is worth anything.
3. **Score the RQ2 event-layer ablation.** `outputs/rq2_v2_on/template.csv` and
   `outputs/rq2_v2_off/template.csv` each hold 150 legs with **0/150 Apple Maps ETAs filled** — 300
   manual lookups, then two `--apple-maps` runs and an on-vs-off comparison. This is the single
   biggest open routing task.
4. **Resolve the event-layer thesis question** (see Known limitations). It gates the headline number.
5. **Run the `--lr-context 1e-4` experiment** and **decide weighted vs unweighted loss.**
6. **Fill the missing baseline lookups** in `outputs/appendix/template.csv` — only 28/50
   `baseline_apple_eta_seconds` and 47/50 `antroute_apple_eta_seconds` are filled, which is why
   `r_squared` has n = 18.
7. **Write down RQ1/RQ2/RQ3.** The repo defines **none of them in prose** — RQ2 is referenced only in
   code comments (`route_trials.py:12`, `evaluate_routing.py:143`) and RQ1/RQ3 appear nowhere. The
   script-to-RQ mapping exists only in the thesis document.
8. **Decide the fate of the old per-node pipeline** (`traffic_risk_model.py` / `train_stgnn.py` /
   `predict_risk.py`) — retire it or document it as an ablation. Still undecided after two sessions,
   and it still carries tests.
9. Fix the README's two empty code blocks and the dead `frontend/.env.local` tunnel URL.

## Where things live

- Frames: `MMDA_FRAMES_PILOT` (May 4), `MMDA_FRAMES` (May 4/6/8/11/13/15), `MMDA_FRAMES_B`
  (May 18/20/22/25) — all on Drive, `G:\My Drive\...` locally / `/content/drive/MyDrive/...` in Colab.
- Checkpoints: Drive `MMDA_CHECKPOINTS/` — **not in the repo.** v6 is the one behind the served scores.
- Served risk artefact: `data/processed/risk_scores/risk_edges.csv` (120 MB, **gitignored — must be
  hand-copied**) + `risk_summary.json` + `risk_calibration.json`. `GET /health/model` exists
  specifically so "which model is this?" is answerable during a demo.
- Routing results: `outputs/routing_eval/` (appendix1/2/3, `metrics.json`, `per_trip.csv`).
  Apple Maps sheets: `outputs/appendix/`.
- Camera → intersection map: `configs/camera_nodes.csv` (`cctv_locations.csv` is stale).
- Training/scoring notebook: `notebooks/train_and_score_colab.ipynb` — still runs
  `RUN = "stgnn_edge_thesis"` with `--split auto` and carries stale 18-session outputs. **It does not
  produce v6 or v7; don't cite its displayed metrics.** Always re-open fresh from GitHub after a
  notebook-level fix rather than trusting an open tab.
- Tests: `traffic_system/tests/` — run from `traffic_system/`, but install
  `requirements.txt` first (see "How to run it").

## Repo hygiene (in progress on this branch)

A cleanup pass is underway for code review. Already deleted in the working tree:
`frontend/components/styles.js` (0 bytes, unused) and two orphaned `.pyc` files whose sources
(`gdelt_scraper.py`, `news_scraper.py`) no longer exist.

Still flagged, not yet removed:
- **Unused**: `src/vision/cctv_extractor.py`, `src/vision/offline_extractor.py`,
  `src/vision/mmda_cctv_pipeline.py` — three superseded extraction prototypes, zero references
  anywhere; the last hardcodes a personal local path.
- **Stale snapshots**: `outputs/appendix/lookups.round1.csv`, `lookups.before_eta.csv`
  (canonical is `lookups.csv`).
- **Committed despite `.gitignore`**: 8 remaining `__pycache__/*.pyc`, `twitter_collection.log(.err)`,
  `data/processed/embedding_run(.error).log`, and 2 hash-named files in `src/ingestion/cache/`.
- **Worth a team decision**: `data/processed/embeddings.pt` is a tracked large binary matching an
  ignored pattern, and `outputs/**` commits experiment results straight into source control.
