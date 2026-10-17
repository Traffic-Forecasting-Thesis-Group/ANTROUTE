"""Inventory actual local evidence without certifying it or filling missing records."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.routing.network import CITIES, FEATURES

ROOT = Path(__file__).resolve().parents[1]


def weather_inventory(path):
    if not path.exists():
        return {"available": False}
    dates, timestamps, grids = set(), set(), set()
    count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            count += 1
            dates.add(row["date"])
            if row.get("hour") is not None:
                timestamps.add((row["date"], int(row["hour"])))
            grids.add(row.get("grid_cell"))
    return {"available": True, "rows": count, "first_date": min(dates), "last_date": max(dates),
            "distinct_dates": len(dates), "distinct_hours": len(timestamps), "grid_cells": len(grids),
            "source_type_and_station_provenance_verified": False}


def audit():
    policy = json.loads((ROOT / "configs/historical_study.json").read_text(encoding="utf-8"))
    start, end = (datetime.fromisoformat(policy[key]) for key in ("start_inclusive", "end_exclusive"))
    required_hours = int((end - start).total_seconds() / 3600)
    if policy["allow_estimates"]:
        from app.study_policy import study_status
        from src.routing.estimation import estimation_registry
        assumptions = json.loads((ROOT / "configs/study_assumptions.json").read_text())
        return {"audited_at": datetime.now(timezone.utc).isoformat(), "ready": study_status()["ready"],
                "period": policy, "cities": list(CITIES), "required_hours_per_entity": required_hours,
                "feature_audit": {feature: {"status": entry["classification"], "local_evidence": entry["basis"],
                                            "reference": entry["source"], "required_evidence": "Estimates authorized; not independently validated historical measurements"}
                                  for feature, entry in estimation_registry(assumptions).items()},
                "weather_files": {name: weather_inventory(ROOT / "data/processed/temporal" / name)
                                  for name in ("weatherstack_historical.csv", "weatherstack_hourly.csv")},
                "note": "Release readiness means the disclosed estimation pipeline is operational, not that historical traffic was measured or estimates calibrated."}
    needed = {
        "distance": "Dated road geometry/length inventories and validity intervals covering 2021–2026; documented geometric calculation is allowed, guessing historical roads is not.",
        "traffic_congestion": "Source-verified hourly segment traffic observations or a documented congestion measure calculated from actual observations.",
        "travel_time": "Observed hourly segment traversal times with timestamps, sample counts, and spatial/directional coverage.",
        "number_of_intersections": "Dated junction inventories or counts derived from historically valid mapped geometry.",
        "traffic_signal": "Dated signal inventories, installation/removal records, and confirmed absence where applicable; missing tags are not zero.",
        "road_type": "Dated road classifications and effective intervals; current classification cannot be copied backward.",
        "time_of_day": "Asia/Manila hour derived from each real timestamp, separate from congestion; no synthetic hourly penalty profile.",
        "weather": "Hourly observations with station/provider evidence and documented coverage of the five cities for the complete period.",
        "flood_risk": "A defined non-estimated feature supported by dated evidence; observed flood records and coverage cannot be substituted by a return-period simulation.",
        "road_capacity": "Dated, source-documented road capacity records and effective intervals; no assumed vehicles-per-hour-per-lane constants or guessed lanes.",
    }
    local = {
        "distance": "September 2026 OSMnx graph; outside the study period and without historical validity intervals.",
        "traffic_congestion": "Only the archived synthetic hourly scenario profile; ineligible.",
        "travel_time": "Only the archived distance/speed/congestion calculation; ineligible as observed travel time.",
        "number_of_intersections": "Derived September 2026 graph counts; no historical coverage.",
        "traffic_signal": "September 2026 OSM tags; unrecorded locations were assumed absent in the prototype.",
        "road_type": "September 2026 OSM tags; no period-specific inventory.",
        "time_of_day": "Only timestamps in the limited local weather files; no complete linked historical routing dataset.",
        "weather": "Local provider CSVs cover June 2025–May 2026 only; observation provenance and city coverage still require verification. October 2026 current observations are out of period.",
        "flood_risk": "Retained NOAH five-year-return-period hazard polygons are simulation-based, not a five-year hourly observation archive.",
        "road_capacity": "Archived assumed 1,500 vehicles/hour/lane and tagged-or-assumed lane counts; ineligible.",
    }
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(), "ready": False,
        "period": policy, "cities": list(CITIES), "required_hours_per_entity": required_hours,
        "feature_audit": {feature: {"status": "not_verified_for_full_period", "local_evidence": local[feature],
                                     "required_evidence": needed[feature]} for feature in FEATURES},
        "weather_files": {name: weather_inventory(ROOT / "data/processed/temporal" / name)
                          for name in ("weatherstack_historical.csv", "weatherstack_hourly.csv")},
        "note": "Date extents alone do not prove complete city/road-hour coverage or non-estimated source values. No imputation, backfilling, synthetic profiles, or copying present-day data into past years is permitted.",
    }


def main():
    result = audit()
    output = ROOT / "data/quality/historical_input_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Estimation release ready={result['ready']}; {result['required_hours_per_entity']} hourly timestamps; see the per-feature classifications.")
    print("Saved data/quality/historical_input_audit.json")


if __name__ == "__main__":
    main()
