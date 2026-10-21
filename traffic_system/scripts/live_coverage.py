"""Report how much of the study network a live HERE snapshot covers.

Matches the flow (and road closures) of one saved snapshot to the 30,511 directed segments and prints the
share of segments and of road length that receive a live speed, overall and by road type. Run it on the
first snapshot to see whether HERE covers the network well enough before relying on it.

Usage (from traffic_system):
  python -m scripts.live_coverage                   latest saved flow snapshot
  python -m scripts.live_coverage --file <path>     a specific *_flow.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live.here import flow_items, incident_items  # noqa: E402
from src.live.conditions import load_assumptions, usable_flow  # noqa: E402
from src.live.matching import coverage, load_segments, match  # noqa: E402

RAW = ROOT / "data/live/raw"
SEGMENTS = ROOT / "data/processed/features/road_segments.csv"


def read(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path)
    args = parser.parse_args(argv)
    flow_path = args.file or max(RAW.glob("*/*_flow.json.gz"), default=None)
    if flow_path is None:
        sys.exit("No saved flow snapshot; run python -m scripts.collect_live --once first.")
    record = read(flow_path)
    items = usable_flow(flow_items(record["payload"]), load_assumptions())
    segments = load_segments(SEGMENTS)
    matched = match(segments, items)
    report = coverage(segments, matched)
    ratios = [items[i]["speed_mps"] / items[i]["free_flow_mps"] for i in matched.values()]

    incidents_path = flow_path.with_name(flow_path.name.replace("_flow", "_incidents"))
    closures = []
    if incidents_path.exists():
        closures = [item for item in incident_items(read(incidents_path)["payload"]) if item["road_closed"]]
    closed = match(segments, closures, min_share=0.4) if closures else {}

    total = report["total"]
    print(f"Snapshot {record['captured_at_manila']}: {len(items)} live flow lines, {len(closures)} road closures")
    print(f"Matched {total['matched']:,} of {total['segments']:,} segments ({total['segment_share']:.1%}), "
          f"{total['matched_km']:.1f} of {total['km']:.1f} km ({total['length_share']:.1%})")
    print(f"{'road type':<16}{'segments':>10}{'matched':>10}{'by length':>11}")
    for road_type, row in report["by_road_type"].items():
        print(f"{road_type:<16}{row['segments']:>10,}{row['segment_share']:>10.1%}{row['length_share']:>11.1%}")
    if ratios:
        print(f"Speed / free-flow on matched segments: median {statistics.median(ratios):.2f}, "
              f"min {min(ratios):.2f}, max {max(ratios):.2f}")
    print(f"Segments under a reported closure: {len(closed):,}")
    out = ROOT / "data/live" / f"coverage_{flow_path.parent.name}_{flow_path.name[:4]}.json"
    out.write_text(json.dumps({"snapshot": record["captured_at_manila"], "flow_lines": len(items),
                               "closures": len(closures), "closed_segments": len(closed), **report}, indent=2),
                   encoding="utf-8")
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
