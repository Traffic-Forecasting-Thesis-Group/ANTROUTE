"""
Check — and correct — which road junction each CCTV camera is pinned to.

    python scripts/check_camera_siting.py
    python scripts/check_camera_siting.py --set "EDSA-Kamuning" 14.6360 121.0430
    python scripts/check_camera_siting.py --set "EDSA-Kamuning" 14.6360 121.0430 --apply

Listing prints every camera with a Google Maps link to the exact node it sits on. Open the
link, compare it with what that camera's footage actually shows, and if they disagree, --set
the junction's real coordinates: the nearest graph node is found and written for you.

This matters more than it looks. A camera's node decides which road edges receive its
Light/Medium/Heavy labels, so a camera pinned to the wrong junction teaches the model about
a road it cannot see. Two were found wrong here: EDSA-Aurora sat 10.8 km away at EDSA x Taft
in Pasay, and EDSA-Ortigas-Shaw sat on Ortigas Ave while the camera looks at Shaw Blvd,
1.4 km apart. Both were silently corrupting training targets and every risk score derived
from them.

The source of truth is the is_cctv_node flag in full_network_static_features.csv -- NOT
key_intersections_node_order.csv, which looks authoritative but is a derived file that
nothing reads. Both are updated here so they cannot drift apart.

After any change the k-hop subgraph changes shape, so data/processed/risk_scores/
risk_edges.csv must be regenerated before its numbers mean anything again.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SPATIAL = REPO_ROOT / "data" / "processed" / "spatial"
FEATURES = SPATIAL / "full_network_static_features.csv"
KEY_ORDER = SPATIAL / "key_intersections_node_order.csv"
CAMERA_MAP = REPO_ROOT / "configs" / "camera_nodes.csv"


def camera_ids_by_intersection() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if CAMERA_MAP.exists():
        with CAMERA_MAP.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                out.setdefault(row["intersection"].strip(), []).append(row["camera_id"].strip())
    return out


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def show(features: pd.DataFrame) -> None:
    cameras = features[features["is_cctv_node"] == True]  # noqa: E712 (pandas mask, not a bool test)
    ids = camera_ids_by_intersection()
    print(f"{len(cameras)} cameras pinned in {FEATURES.name}\n")
    for _, row in cameras.sort_values("cctv_label").iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        units = ids.get(str(row["cctv_label"]), [])
        print(f"  {row['cctv_label']}")
        print(f"      node {int(row['node_id'])}   {lat:.6f}, {lon:.6f}")
        print(f"      map  https://www.google.com/maps?q={lat:.6f},{lon:.6f}")
        print(f"      feed {', '.join(units) if units else '(no camera id in camera_nodes.csv)'}")
        print()
    print("Open each map link and compare it with that camera's own footage. If they show")
    print("different junctions, re-pin it:")
    print('  python scripts/check_camera_siting.py --set "<label>" <lat> <lon>        # preview')
    print('  python scripts/check_camera_siting.py --set "<label>" <lat> <lon> --apply')


def repin(features: pd.DataFrame, label: str, lat: float, lon: float, apply: bool) -> None:
    current = features[features["cctv_label"].astype(str) == label]
    if current.empty:
        labels = sorted(features.loc[features["is_cctv_node"] == True, "cctv_label"].astype(str))  # noqa: E712
        raise SystemExit(f"no camera labelled {label!r}. Known labels: {labels}")

    distances = haversine_km(features["lat"].to_numpy(), features["lon"].to_numpy(), lat, lon)
    nearest = int(np.argmin(distances))
    target = features.iloc[nearest]
    old = current.iloc[0]
    moved = float(haversine_km(float(old["lat"]), float(old["lon"]), lat, lon))

    print(f"{label}")
    print(f"  now at node {int(old['node_id'])}  {float(old['lat']):.6f}, {float(old['lon']):.6f}")
    print(f"  you say     {lat:.6f}, {lon:.6f}   ({moved:.2f} km away)")
    print(f"  nearest node {int(target['node_id'])}  {float(target['lat']):.6f}, {float(target['lon']):.6f}"
          f"   ({distances[nearest] * 1000:.0f} m from the point you gave)")

    if distances[nearest] > 0.5:
        print("\n  WARNING: the nearest road node is over 500 m from that point. Check the "
              "coordinates -- the junction may not be on this road network at all.")
    if not apply:
        print("\n  preview only; pass --apply to write it")
        return

    features.loc[features["node_id"] == int(old["node_id"]), ["is_cctv_node", "cctv_label"]] = [False, None]
    features.loc[features["node_id"] == int(target["node_id"]), ["is_cctv_node", "cctv_label"]] = [True, label]
    features.to_csv(FEATURES, index=False)

    if KEY_ORDER.exists():   # derived, but kept in step so the two files never disagree
        key = pd.read_csv(KEY_ORDER)
        key.loc[key["cctv_label"] == label, "node_id"] = int(target["node_id"])
        key.to_csv(KEY_ORDER, index=False)

    print(f"\n  written. Re-run scripts/build_text_placement.py, then regenerate "
          f"risk_edges.csv -- it was scored against the old graph.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs=3, metavar=("LABEL", "LAT", "LON"),
                        help="re-pin a camera to the node nearest these coordinates")
    parser.add_argument("--apply", action="store_true", help="write the change (default: preview)")
    args = parser.parse_args()

    features = pd.read_csv(FEATURES)
    if args.set:
        repin(features, args.set[0], float(args.set[1]), float(args.set[2]), args.apply)
    else:
        show(features)


if __name__ == "__main__":
    main()
