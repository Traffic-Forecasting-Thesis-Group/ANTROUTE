"""Date-indexed provider records; spatial assignment is an explicit approximation."""
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from shapely.geometry import shape

from .network import CITIES
from .weather import observed_effect

ROOT = Path(__file__).resolve().parents[2]


def build_weather(boundaries, output, nodes=None):
    source = ROOT / "data/processed/temporal/weatherstack_hourly.csv"
    grids = set()
    with source.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            grids.add(row["grid_cell"])
    points = {grid: tuple(int(part) * .05 for part in grid.split("_")) for grid in grids}
    selected = {}
    for city, geometry in boundaries.items():
        center = shape(geometry).representative_point()
        selected[city] = min(points, key=lambda grid: (points[grid][0] - center.y) ** 2 + (points[grid][1] - center.x) ** 2)
    records = {city: {} for city in CITIES}
    grid_records = {grid: {} for grid in sorted(grids)}
    with source.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if not "2025-06-01" <= row["date"] <= "2026-05-31":
                continue
            key = f"{row['date']}T{int(row['hour']):02d}:00"
            grid = row["grid_cell"]
            if key in grid_records[grid]:
                raise ValueError(f"Duplicate weather record for {grid}: {key}")
            record = {"precip_mm": float(row["ws_precip_mm"]), "description": row["ws_description"], "grid_cell": grid}
            grid_records[grid][key] = record
            for city, grid in selected.items():
                if row["grid_cell"] == grid:
                    if key in records[city]:
                        raise ValueError(f"Duplicate weather record for {city}: {key}")
                    records[city][key] = record
    expected = {(datetime(2025, 6, 1) + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:00") for hour in range(8760)}
    for city in CITIES:
        if set(records[city]) != expected:
            raise ValueError(f"Incomplete hourly provider weather coverage for {city}")
    for grid, data in grid_records.items():
        if set(data) != expected:
            raise ValueError(f"Incomplete hourly provider weather coverage for grid {grid}")
    node_grids = {}
    for node in nodes or []:
        node_grids[node.id] = min(sorted(points), key=lambda grid:
            (points[grid][0] - node.latitude) ** 2 +
            ((points[grid][1] - node.longitude) * math.cos(math.radians(node.latitude))) ** 2)
    bundle = {"source": "Weatherstack historical CSV; measurement provenance not independently verified",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "spatial_method": "Nearest retained 0.05-degree grid to each road source node; city summaries use representative points. Not road-level measurements.",
              "city_grid_assignment": selected, "hours_per_city": {city: len(data) for city, data in records.items()}, "records": records,
              "records_by_grid": grid_records, "node_grid_assignment": node_grids,
              "grid_count": len(grid_records), "assigned_grid_count": len(set(node_grids.values()))}
    output.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    return bundle


@lru_cache(maxsize=1)
def _load(path, modified):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def historical_snapshot(day, hour, network):
    path = ROOT / "data/processed/historical_weather.json"
    bundle = _load(str(path), path.stat().st_mtime_ns)
    key = f"{day.isoformat()}T{hour:02d}:00"
    scenarios = network.provenance["assumptions"]["weather_scenarios"]
    effects, cities = {}, {}
    for city in CITIES:
        record = bundle["records"][city].get(key)
        if record is None:
            raise ValueError(f"No historical weather record for {city} at {key}")
        raw = {"current": {"precip": record["precip_mm"], "weather_descriptions": [record["description"]]}}
        effects[city] = observed_effect(raw, scenarios)
        cities[city] = {"localtime": key.replace("T", " "), "description": [record["description"]],
                        "precip_mm": record["precip_mm"], "grid_cell": record["grid_cell"], "condition": "historical", "retrieved_at": "historical CSV"}
    grid_effects = {}
    for grid in set(bundle.get("node_grid_assignment", {}).values()):
        record = bundle["records_by_grid"][grid].get(key)
        if record is None:
            raise ValueError(f"No historical weather for grid {grid} at {key}")
        grid_effects[grid] = observed_effect({"current": {"precip": record["precip_mm"],
                                "weather_descriptions": [record["description"]]}}, scenarios)
    for node in network.nodes:
        grid = bundle.get("node_grid_assignment", {}).get(node.id)
        if grid is not None:
            effects[node.id] = grid_effects[grid]
    return effects, {"source": bundle["source"], "spatial_method": bundle["spatial_method"],
                     "stale": False, "age_seconds": 0, "cities": cities,
                     "grid_count": bundle.get("grid_count"), "assigned_grid_count": bundle.get("assigned_grid_count")}


def historical_grid_assignments():
    path = ROOT / "data/processed/historical_weather.json"
    return _load(str(path), path.stat().st_mtime_ns).get("node_grid_assignment", {})


def historical_grid_records(day, hour):
    path = ROOT / "data/processed/historical_weather.json"
    bundle = _load(str(path), path.stat().st_mtime_ns)
    key = f"{day.isoformat()}T{hour:02d}:00"
    return {grid: records[key] for grid, records in bundle.get("records_by_grid", {}).items()}
