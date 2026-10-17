"""Export resumable daily segment-hour tables of all ten feature values and path costs.

Example: python -m scripts.export_dataset --start 2025-06-01 --end 2025-06-01
Omit the date options to materialize the complete study year (267M+ rows).
Per-segment static inputs are already in data/processed/features; this joins them with
the hourly weather and traffic tables exactly as routing does.
"""
import argparse
import csv
import gzip
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from app.main import get_network
from app.study_policy import require_historical_release
from src.routing.condition_update import Conditions
from src.routing.datasets import FIELDS, coverage, snapshot_rows

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 6, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 5, 31))
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/segment_hours")
    args = parser.parse_args()
    if not date(2025, 6, 1) <= args.start <= args.end <= date(2026, 5, 31):
        parser.error("Choose an inclusive range within June 2025–May 2026")
    require_historical_release()
    network = get_network()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "dataset_manifest.json"
    modules = ["src/routing/dynamic_weight.py", "src/routing/datasets.py", "src/routing/historical_weather.py",
               "src/routing/historical_traffic.py", "scripts/export_dataset.py"]
    definition = {"source_artifacts": json.loads((ROOT / "data/processed/study_release.json").read_text())["artifacts"],
                  "implementation": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in modules}}
    fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            parser.error("This folder was exported from a different release; choose a separate --output folder.")
    else:
        manifest = {"fingerprint": fingerprint, "definition": definition, "coverage": coverage(network),
                    "fields": FIELDS, "completed_days": {}}
    atomic_json(manifest_path, manifest)
    day = args.start
    while day <= args.end:
        name = f"segment_hours_{day}.csv.gz"
        path = out / name
        previous = manifest["completed_days"].get(str(day))
        if previous and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == previous["sha256"]:
            print(f"Verified existing partition: {day}", flush=True)
            day += timedelta(days=1)
            continue
        temporary = path.with_suffix(".tmp")
        count = 0
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for hour in range(24):
                for row in snapshot_rows(network, Conditions(departure_date=day, time_of_day=hour, weather="historical")):
                    writer.writerow(row)
                    count += 1
        if count != len(network.roads) * 24:
            raise RuntimeError("Incomplete day; partition not published")
        temporary.replace(path)
        manifest["completed_days"][str(day)] = {"file": name, "rows": count,
                                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest["materialized_segment_hour_rows"] = sum(item["rows"] for item in manifest["completed_days"].values())
        manifest["full_year_materialized"] = len(manifest["completed_days"]) == 365
        atomic_json(manifest_path, manifest)
        print(f"Exported {day}: {count:,} rows ({path.stat().st_size / 1024**2:.1f} MiB)", flush=True)
        day += timedelta(days=1)
    print(f"Dataset folder: {out}", flush=True)


if __name__ == "__main__":
    main()
