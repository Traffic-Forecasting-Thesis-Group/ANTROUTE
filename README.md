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
.\venv\Scripts\python -m scripts.preprocess_traffic
.\venv\Scripts\python -m scripts.build_study_network
.\venv\Scripts\python -m scripts.audit_historical_inputs
.\venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The build works offline with the retained road/flood inputs, sourced city boundaries, historical weather CSV, and assumption settings. It writes a scoped network, date-indexed provider weather, estimation registry, summary, and release manifest. Large generated network/weather bundles are gitignored and already exist locally. Rebuild after changing policy or assumption files; hash mismatches block routing.

In a second terminal:

```powershell
cd frontend
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev -- --host 127.0.0.1
```

Open **http://127.0.0.1:5173/**. The frontend connects to the backend on port 8000. Existing Maps credentials are preserved; a key-free boundary/route preview is available. No TomTom, HERE, or new weather API requests are required to use the prepared release.

Choose an origin and destination, a study date, local hour, and historical weather or an explicit scenario. Optional map-selected closures are hypothetical conditions, not verified historical closure records. Calculate the ACO route, inspect the four evaluation items, and expand **Where each feature and estimate comes from** for the full basis.

## Verify

```powershell
cd traffic_system
.\venv\Scripts\python -m pytest tests -q
.\venv\Scripts\python -m scripts.validate_system
.\venv\Scripts\python -m scripts.estimate_sensitivity
cd ../frontend
npm run build
npm run test:e2e
```

Real-network functional results and demand sensitivity appear under `traffic_system/outputs/`. The latter varies assumed demand by -20%, baseline, and +20%, keeping the ACO seed fixed. Neither report establishes empirical historical accuracy or global route optimality. No evaluation metrics beyond the requested four are introduced.

The active release is rebuilt with documented estimation methods; the earlier prototype remains archived. The previously required five-year, estimates-prohibited policy is superseded by the current one-year estimation authorization.
