# ANTROUTE

ACO routing for **San Juan, Manila, Mandaluyong, Makati, and Pasig**, with hourly modeled snapshots during **June 1, 2025–May 31, 2026**. Estimates are permitted and explicitly disclosed.

The system uses exactly the requested ten features: distance, traffic congestion, travel time, number of intersections, traffic signal, road type, time of day, weather, flood risk, and road capacity. Dynamics are time of day, manual road-closure scenarios, and weather. Evaluations are distance traveled, path cost, computational time, and convergence rule.

**This is not a fully measured historical traffic archive.** Traffic/capacity/travel-time/weather effects are modeled. Historical provider weather has 8,760 timestamps for each assigned city grid. The September 2026 road inventory and static NOAH hazards are temporal proxies. Missing signal tags are assumed absent. All caveats are visible in the app.

Read [input feature preprocessing](traffic_system/docs/feature_preprocessing.md), [estimation methods and sources](traffic_system/docs/estimation_methods.md), [architecture](traffic_system/docs/aco_system_architecture.md), and [network/API contract](traffic_system/docs/network_data.md). The executable parameters and literature links are in [study_assumptions.json](traffic_system/configs/study_assumptions.json). Formulas from published references are distinguished from manually chosen, uncalibrated inputs.

## Build and run

From `traffic_system` with the Python environment installed:

```powershell
.\venv\Scripts\python -m pip install -r requirements.txt
.\venv\Scripts\python -m scripts.preprocess_road_features
.\venv\Scripts\python -m scripts.preprocess_traffic          # per-road hourly traffic, ~16 GB, ~6 min
.\venv\Scripts\python -m scripts.build_study_network
.\venv\Scripts\python -m scripts.audit_historical_inputs
.\venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The build works offline with the retained road/flood inputs, sourced city boundaries, historical weather CSV, and assumption settings. It writes a scoped network, date-indexed provider weather, the per-road hourly traffic table (`traffic_hourly.csv`, 30,511 segments × 8,760 hours) with its index, estimation registry, summary, and release manifest. Large generated files are gitignored and must be rebuilt on a fresh checkout. Rebuild after changing policy or assumption files; hash mismatches block routing.

Place search covers road names plus 4,006 named OSM places (barangays, malls, schools, stations, landmarks) in `data/raw/study/places_gazetteer.csv`. To refresh it, download `https://download.geofabrik.de/asia/philippines-latest.osm.pbf` and run `.\venv\Scripts\python -m scripts.fetch_places --extract <path to .pbf>`.

In a second terminal:

```powershell
cd frontend
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev -- --host 127.0.0.1
```

Open **http://127.0.0.1:5173/**. The frontend connects to the backend on port 8000. Existing Maps credentials are preserved; a key-free boundary/route preview is available. No TomTom, HERE, or new weather API requests are required to use the prepared release.

Choose an origin and destination, a study date, local hour, and historical weather or an explicit scenario. Optional map-selected closures are hypothetical conditions, not verified historical closure records. Calculate the ACO route, inspect the four evaluation items, and expand **Where each feature and estimate comes from** for the full basis.

## Algorithms and dynamic experiment

- `POST /routes/plan`: ANTROUTE ACO on the ten features with CRITIC weights, one date/hour snapshot.
- `POST /routes/baseline`: Cheng (2023) improved ACO baseline on the same snapshot.
- `POST /routes/trip`: a simulated trip with `algorithm` = `antroute` (proposed) or `cheng` (baseline). Conditions change as the trip crosses into the next hour, and `events` can close/reopen roads or change the weather at a given minute.

Answer the statement of the problem (writes `outputs/sop_runs.csv`, `outputs/sop_results.json` and the readable `outputs/sop_report.md`):

```powershell
.env\Scripts\python -m scripts.evaluate_sop --trips 10 --seeds 2
```

- SOP 1: distance travelled and path cost under five traffic conditions (off-peak, morning and evening peak onset, road closure, heavy-rain onset).
- SOP 2: computational time, initial routing time, dynamic re-routing time, and convergence rate (N_conv: the iteration at which the best path cost had not improved for 20 consecutive iterations; both algorithms stop on this rule, and reaching the iteration limit first is recorded as not converged).
- SOP 3: ANTROUTE vs Cheng on distance, path cost, computational time and convergence rate, paired by trip and condition with d = baseline − ANTROUTE; two-sided Wilcoxon signed-rank test (W = min(W⁺, W⁻)) with Holm-adjusted p-values across the four measures (α = 0.05). Add `--from-runs` to re-analyse `outputs/sop_runs.csv` without re-running trips.

See [architecture](traffic_system/docs/aco_system_architecture.md#algorithms) for how the three differ.

## Verify

```powershell
cd traffic_system
.env\Scripts\python -m pytest tests -q
.env\Scripts\python -m scripts.validate_system
.env\Scripts\python -m scripts.estimate_sensitivity
cd ../frontend
npm run build
npm run test:e2e
```

Real-network functional results and demand sensitivity appear under `traffic_system/outputs/`. The latter varies assumed demand by -20%, baseline, and +20%, keeping the ACO seed fixed. Neither report establishes empirical historical accuracy or global route optimality. No evaluation metrics beyond the requested four are introduced.

The active release is rebuilt with documented estimation methods; the earlier prototype remains archived. The previously required five-year, estimates-prohibited policy is superseded by the current one-year estimation authorization.
