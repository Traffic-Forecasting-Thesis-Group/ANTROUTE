"""Preprocess the hourly traffic table, the traffic counterpart of weatherstack_hourly.csv.

Input:  data/processed/features/road_segments.csv (city, road type, free-flow time and geometry of every segment)
        configs/study_assumptions.json (hourly profile, road-type factors, BPR, variation)
Output: data/processed/temporal/traffic_hourly.csv
        one row per road segment per hour, June 1, 2025 - May 31, 2026:
        date, hour, road_id, traffic_congestion, travel_time_factor, travel_time_s
        data/processed/temporal/traffic_hourly.index.json (byte offset of every hour, row count, SHA-256)

Values are estimated, not observed. travel_time_s is the OSM free-flow time multiplied by
travel_time_factor, before any weather adjustment; traffic_congestion is the congestion feature.
The full table is large (about 15 GB); it is gitignored and reproducible from the inputs.

Run: python -m scripts.preprocess_traffic   (after scripts.preprocess_road_features)
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from shapely import wkt

from src.routing.historical_traffic import HOURS, TABLE, estimated_hours, write_table

ROOT = Path(__file__).resolve().parents[1]


def segment_inputs(path):
    """Rows the traffic model needs, sorted by road_id; location is the segment midpoint."""
    roads = []
    with Path(path).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            middle = wkt.loads(row["geometry_wkt"]).interpolate(0.5, normalized=True)
            roads.append({"road_id": row["road_id"], "city": row["city"], "road_type": row["road_type"],
                          "free_flow_time_s": float(row["free_flow_time_s"]),
                          "longitude": middle.x, "latitude": middle.y})
    return sorted(roads, key=lambda road: road["road_id"])


def progress(hourly):
    for number, item in enumerate(hourly, 1):
        if number % 240 == 0 or number == HOURS:
            print(f"Traffic table: {number}/{HOURS} hours", flush=True)
        yield item


def main():
    assumptions = json.loads((ROOT / "configs/study_assumptions.json").read_text(encoding="utf-8"))
    roads = segment_inputs(ROOT / "data/processed/features/road_segments.csv")
    checks = write_table(ROOT / TABLE, progress(estimated_hours(roads, assumptions)), {
        "method": "BPR on an assumed per-road demand-to-capacity index",
        "traffic_variation": assumptions["traffic_variation"]})
    if checks["hours"] != HOURS or checks["roads"] != len(roads):
        raise ValueError("Incomplete hourly traffic coverage")
    quality = ROOT / "data/quality/feature_preprocessing.json"
    existing = json.loads(quality.read_text(encoding="utf-8")) if quality.exists() else {}
    existing["traffic"] = {"generated_at": datetime.now(timezone.utc).isoformat(), "table": TABLE,
                           "classification": "estimated_not_observed", "rows": checks["rows"],
                           "road_segments": checks["roads"], "hours_per_segment": checks["hours"],
                           "bytes": checks["bytes"], "sha256": checks["sha256"],
                           "no_duplicates": True, "values_in_range": True}
    quality.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Traffic table: {checks['rows']:,} rows ({checks['roads']:,} segments x {HOURS} hours, "
          f"{checks['bytes'] / 1e9:.1f} GB) -> {TABLE}", flush=True)


if __name__ == "__main__":
    main()
