"""Date-indexed provider records; spatial assignment is an explicit approximation."""
import csv
import hashlib
import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from shapely.geometry import shape

from .network import CITIES
from .weather import observed_effect

ROOT = Path(__file__).resolve().parents[2]


def build_weather(boundaries, output):
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
    with source.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if not "2025-06-01" <= row["date"] <= "2026-05-31":
                continue
            key = f"{row['date']}T{int(row['hour']):02d}:00"
            for city, grid in selected.items():
                if row["grid_cell"] == grid:
                    if key in records[city]:
                        raise ValueError(f"Duplicate weather record for {city}: {key}")
                    records[city][key] = {"precip_mm": float(row["ws_precip_mm"]), "description": row["ws_description"], "grid_cell": grid}
    expected = {(datetime(2025, 6, 1) + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:00") for hour in range(8760)}
    for city in CITIES:
        if set(records[city]) != expected:
            raise ValueError(f"Incomplete hourly provider weather coverage for {city}")
    bundle = {"source": "Weatherstack historical CSV; measurement provenance not independently verified",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "spatial_method": "Nearest 0.05-degree grid to city representative point; not road-level weather measurement",
              "city_grid_assignment": selected, "hours_per_city": {city: len(data) for city, data in records.items()}, "records": records}
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
    return effects, {"source": bundle["source"], "spatial_method": bundle["spatial_method"],
                     "stale": False, "age_seconds": 0, "cities": cities}
