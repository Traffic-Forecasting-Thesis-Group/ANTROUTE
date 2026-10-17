"""Preprocess the hourly traffic table, the traffic counterpart of weatherstack_hourly.csv.

Input:  data/processed/features/road_segments.csv (city and road type of every segment)
        configs/study_assumptions.json (hourly profile, road-type factors, BPR, variation)
Output: data/processed/temporal/traffic_hourly.csv
        one row per location (city x road type) per hour, June 1, 2025 - May 31, 2026:
        date, hour, location_id, city, road_type, day_type, traffic_congestion, travel_time_factor, source

Values are estimated, not observed. Each segment's travel time is its OSM free-flow time
multiplied by travel_time_factor; traffic_congestion is used as the congestion feature.

Run: python -m scripts.preprocess_traffic   (after scripts.preprocess_road_features)
"""
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.routing.historical_traffic import FIELDS, HOURS, TABLE, table_rows

ROOT = Path(__file__).resolve().parents[1]


def validate(path, locations):
    hours, keys = Counter(), set()
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["date"], row["hour"], row["location_id"])
            if key in keys:
                raise ValueError(f"Duplicate traffic record: {key}")
            keys.add(key)
            if not 0 <= float(row["traffic_congestion"]) < 1 or float(row["travel_time_factor"]) < 1:
                raise ValueError(f"Traffic value out of range: {key}")
            hours[row["location_id"]] += 1
    incomplete = [location for location in locations if hours[location] != HOURS]
    if incomplete or set(hours) != set(locations):
        raise ValueError(f"Incomplete hourly traffic coverage for {', '.join(incomplete) or 'unexpected locations'}")
    return {"rows": len(keys), "locations": len(locations), "hours_per_location": HOURS,
            "no_duplicates": True, "values_in_range": True}


def main():
    assumptions = json.loads((ROOT / "configs/study_assumptions.json").read_text(encoding="utf-8"))
    with (ROOT / "data/processed/features/road_segments.csv").open(encoding="utf-8") as stream:
        locations = {(row["traffic_location"], row["city"], row["road_type"]) for row in csv.DictReader(stream)}
    path = ROOT / TABLE
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table_rows(locations, assumptions))
    checks = validate(temporary, [location for location, _, _ in locations])
    temporary.replace(path)
    quality = ROOT / "data/quality/feature_preprocessing.json"
    existing = json.loads(quality.read_text(encoding="utf-8")) if quality.exists() else {}
    existing["traffic"] = {"generated_at": datetime.now(timezone.utc).isoformat(), "table": TABLE,
                           "classification": "estimated_not_observed", **checks}
    quality.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Traffic table: {checks['rows']:,} rows ({checks['locations']} locations x {HOURS} hours) -> {TABLE}", flush=True)


if __name__ == "__main__":
    main()
