# Input feature preprocessing

Every input feature follows the same path as the Weatherstack weather data: **raw source → cleaned table → validation → routing reads the table**. The router never reads a raw source directly. Study area: San Juan, Manila, Mandaluyong, Makati and Pasig. Period: June 1, 2025 – May 31, 2026, hourly, Asia/Manila.

## Pipeline

Run from `traffic_system`:

```powershell
.\venv\Scripts\python.exe -m scripts.preprocess_road_features   # OSM + NOAH  -> data/processed/features/*.csv
.\venv\Scripts\python.exe -m scripts.preprocess_traffic         # -> data/processed/temporal/traffic_hourly.csv (~16 GB, ~6 min)
.\venv\Scripts\python.exe -m scripts.build_study_network        # tables + weather -> study_network.json, release manifest
```

Each preprocessing step validates its output and writes a summary to `data/quality/feature_preprocessing.json`. The build records SHA-256 hashes of the tables it read; routing is blocked if a release artifact changes without a rebuild.

## Sources and tables

| Feature | Raw source | Preprocessed table | Column(s) |
|---|---|---|---|
| Distance | OSM road graph (OSMnx) | `features/road_segments.csv` | `distance_m` |
| Traffic Congestion | Estimated (no observed source) | `temporal/traffic_hourly.csv` | `traffic_congestion` |
| Travel Time | OSM speed + traffic table + weather | `temporal/traffic_hourly.csv` | `travel_time_s` (= `free_flow_time_s` × `travel_time_factor`, before weather) |
| Number of Intersections | OSM graph | `features/road_nodes.csv` | `is_intersection` (street count ≥ 3) |
| Traffic Signal | OSM `traffic_signals` tags | `features/road_nodes.csv`, `road_segments.csv` | `traffic_signal`, `signal_at_target` |
| Road Type | OSM `highway` tag | `features/road_segments.csv` | `road_type` |
| Time profile | Hourly traffic table, development dates | fitted at build time | `time_profile` (weekday/weekend × 24) |
| Weather | Weatherstack hourly | `temporal/weatherstack_hourly.csv` | `ws_precip_mm`, `ws_description`, … |
| Flood exposure | Project NOAH 5-year hazard polygons | `features/segment_flood.csv` | `flood_km_class1..3`, `flood_coverage` |
| Capacity burden | OSM lanes + signal tags | `features/road_segments.csv` | `road_capacity_vph`, `capacity_method`, `capacity_uncertainty` |

No table contains vehicle volume or vehicle counts. Volume is not a study feature.

### Static tables (`data/processed/features`)

`scripts/preprocess_road_features.py` cleans the OSMnx graph and NOAH polygons:

1. **Scope:** a source edge is kept only if both endpoints and the whole geometry lie inside the five city boundaries. Each node and segment gets the city it lies in; a segment takes its source node's city.
2. **Drivable public roads only:** the edge's OSM `highway` tag must be a configured road type, and `access=private|no` is excluded.
3. **Geometry and distance:** the OSM polyline is used when present, otherwise the endpoint segment (OSMnx convention). Distance is the WGS84 geodesic length; zero-length edges are dropped.
4. **Speed:** OSM `maxspeed` (mph converted), else the road-type default. `free_flow_time_s = distance / speed`.
5. **Lanes:** directional `lanes:forward|backward`, else `lanes` (split in half on two-way roads), else the road-type default.
6. **Capacity:** `K = directional lanes × 1900 (assumed base saturation flow) × g/C`, with an assumed `g/C = 0.5` when the downstream node has a mapped signal and 1 otherwise; method and uncertainty are recorded per segment.
7. **Flood:** the segment is intersected with each NOAH class; overlapping polygons resolve to the highest class, and the geodesic length (km) in each class is stored as `flood_km_class1..3`. `flood_coverage` is `mapped` inside the hazard dataset's extent and `unknown` outside it. The class severity is applied at scoring time (see [routing_contract.md](routing_contract.md)).
8. **Traffic location:** each segment's `traffic_location` is its own `road_id`, the key into the hourly traffic table, so every segment has its own hourly series.

Every value has a `*_source` column (for example `osm_maxspeed` vs `road_type_default_speed`), so defaults are never presented as mapped data. Checks: unique IDs, endpoints exist, positive distance/speed/capacity, known road types, one flood row per segment, flood portions nonnegative and no longer than the segment, declared coverage, all five cities present.

Current output: 12,398 nodes, 30,511 directed segments. 117,877 source edges were outside the five cities, 84 were not public driving roads and 11 had invalid geometry.

### Hourly traffic table (`data/processed/temporal/traffic_hourly.csv`)

This table is laid out like `weatherstack_hourly.csv`, but its values are **estimated**: no observed hourly traffic exists for these cities and this period. TomTom Traffic Stats (historical speeds, access through a partner trial) or MMDA counts (FOI request) could replace this step later without changing the rest of the pipeline.

```
date,hour,road_id,traffic_congestion,travel_time_factor,travel_time_s
2025-06-02,8,12067330:6621954486:0,0.251640,1.336256,19.471
```

One row per **road segment** per hour: 30,511 segments × 8,760 hours = **267,276,360 rows (about 16 GB)**. Rows are ordered by date, hour, then `road_id`. For each row:

- `x` = assumed hourly demand-to-capacity index × road-type factor × weekend factor (0.75) × repeatable variation
  `exp(0.6 × (0.20 × day[city] + 0.25 × area[~1.1 km cell, 3-hour] + 0.35 × road[segment] + 0.25 × road[segment, 3-hour]))`.
  Each term is a uniform value in [−1, 1] from a seeded random stream (seed 42). Three-hour values are interpolated linearly, so traffic changes smoothly from hour to hour. Neighbouring roads share the area term, so congestion is spatially coherent; the per-road terms make each segment distinct. `x` is a demand index, not a vehicle count, and it is not stored.
- `travel_time_factor = 1 + 0.15 x⁴` (BPR); `traffic_congestion = 1 − 1 / travel_time_factor`; `travel_time_s = free_flow_time_s × travel_time_factor` (weather is applied at routing time).

`traffic_hourly.index.json` records the byte offset of every hour, the row count and the table's SHA-256, so routing reads one hour (about 1.9 MB) instead of the whole file. The CSV is gitignored and reproducible; the index is part of the release, and routing is blocked if the CSV's size no longer matches it.

Checks: every segment has all 8,760 hours, no duplicates, congestion in [0, 1), factor ≥ 1. All parameters are in `configs/study_assumptions.json` (`hourly_demand_capacity_ratio`, `road_types.*.traffic_factor`, `weekend_demand_factor`, `traffic_variation`).

## How routing reads the tables

For a request date and hour, the API loads the matching rows of the weather and traffic tables. For each segment:

- `traffic_congestion` = the table value for the segment.
- `travel_time` = `free_flow_time_s × travel_time_factor × weather travel-time multiplier`.
- The remaining eight features come from the static tables and the request.

Closed roads (a dynamic condition) are removed from the graph. The ten values are normalized and weighted into the segment cost that ACO minimizes.

## Exports

- `POST /dataset/snapshot`, `/dataset/export` (CSV) and `/dataset/map` return the ten feature values and path cost of every segment for one date and hour, using the same function as routing. `GET /dataset/info` reports coverage.
- `python -m scripts.export_dataset --start 2025-06-01 --end 2025-06-01` writes daily gzip tables (732,264 rows per day) to `data/processed/segment_hours/`, which is gitignored.
