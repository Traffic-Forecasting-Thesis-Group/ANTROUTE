# Dated estimation release

Period: `[2025-06-01T00:00:00+08:00, 2026-06-01T00:00:00+08:00)`, hourly. Estimates are explicitly allowed and must carry source/method disclosures.

Build the local release with `python -m scripts.build_study_network`. It uses the retained road graph, five-city boundaries, NOAH polygons, historical provider weather CSV, and explicit assumption settings. No paid traffic API or live-weather request is required.

Generated files:

- `data/processed/study_network.json`: scoped road graph, ten-feature values, BPR inputs, and source labels.
- `data/processed/historical_weather.json`: exact city-grid assignments and 8,760 provider records per city; cities may share grids.
- `data/processed/study_release.json`: period policy, weather-hour counts, and artifact hashes.
- `data/processed/estimation_registry.json`: ten-feature formulas, classifications, and references.
- `data/processed/study_network_summary.json`: scope counts and release mode.

The large network/weather bundles are generated and gitignored. The existing archived prototype is not simply copied back; the active release is rebuilt with the documented method.

Example request:

```json
{
  "origin": "3198395473",
  "destination": "243204561",
  "conditions": {
    "departure_date": "2025-06-02",
    "time_of_day": 8,
    "weather": "historical",
    "closed_road_ids": []
  }
}
```

Omitted dates default to June 1, 2025; the response echoes the actual date. Dates outside the study year are rejected. The `historical` option uses provider weather for the selected date/hour. Other weather choices are explicit scenarios. Closures are manually supplied scenario constraints. Each request uses a fixed snapshot and fresh colony.

`GET /study/status` reports release readiness and authorized estimation mode. `GET /conditions/weather?departure_date=2025-06-02&time_of_day=8` shows the provider snapshot and spatial-assignment caveat. Live-weather refresh remains disabled for the historical study.

Cost calculation uses ten features; the time-of-day peak policy is distinct from congestion. Road inventories/flood maps remain static surrogates. See [estimation_methods.md](estimation_methods.md) before interpreting results as historical evidence.
