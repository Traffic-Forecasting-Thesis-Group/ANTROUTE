"""
Slim YOLO vehicle counts for the app: camera_id, timestamp, n_vehicles from the labeling step's
auto_labels.csv files (which also carry every frame's boxes, ~10 MB each).

    python scripts/build_vehicle_counts.py "G:/My Drive/MMDA_FRAMES/auto_labels.csv" \
        "G:/My Drive/MMDA_FRAMES_B/auto_labels.csv" "G:/My Drive/MMDA_FRAMES_PILOT/auto_labels.csv"

Writes data/processed/vehicle_counts/vehicle_counts.csv (gitignored, like the frames it comes
from). The app reads that folder for two things: the "recorded" inputs panel (vehicle counts at
the cameras on a route) and the baseline's traffic-flow term (Cheng eq. 6), the same term
scripts/evaluate_routing.py plans the baseline with. A frame present in two files is kept once.
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("auto_labels", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/vehicle_counts/vehicle_counts.csv")
    a = p.parse_args()
    frames = [pd.read_csv(f, usecols=["camera_id", "timestamp", "n_vehicles"]) for f in a.auto_labels]
    counts = pd.concat(frames, ignore_index=True).drop_duplicates(["camera_id", "timestamp"])
    counts = counts.sort_values(["timestamp", "camera_id"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(a.out, index=False)
    print(f"{len(counts):,} frames, {counts['camera_id'].nunique()} cameras, "
          f"{counts['timestamp'].min()[:10]} to {counts['timestamp'].max()[:10]} -> {a.out}")


if __name__ == "__main__":
    main()
