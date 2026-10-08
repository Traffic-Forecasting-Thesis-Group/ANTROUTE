"""
Work out which camera each gazetteer intersection reports to, and write it to
configs/text_placement.csv.

    python scripts/build_text_placement.py
    python scripts/build_text_placement.py --max-hops 6 --dry-run

Why this exists: a tweet is about a place, and until now text only reached the model when it
named one of the camera intersections themselves. Every report on the corridor *between*
cameras was dropped -- 8 usable places out of the 45 the gazetteer can locate on this subgraph.

The RADR STGNN is a spatial model: its GCN propagates along the road graph, so an incident a few
intersections away genuinely bears on what a camera sees. Attaching a report to the nearest
camera by hop count is that same propagation, done once here instead of being discarded before
the model ever sees it. Hops, not kilometres: two points either side of a river are close in a
straight line and far apart to a driver.

Written to a config rather than computed during training for the same reasons
configs/event_intersections.csv is: building the subgraph is slow, the result is small and
rarely changes, and a reviewer can read the file and check where any report would land.
"""

import argparse
import csv
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.graph_data import DEFAULT_K, build_subgraph  # noqa: E402

INTERSECTIONS_CSV = REPO_ROOT / "configs" / "event_intersections.csv"
CAMERA_MAP_CSV = REPO_ROOT / "configs" / "camera_nodes.csv"
OUT_CSV = REPO_ROOT / "configs" / "text_placement.csv"

SNAP_KM = 0.5  # how close a gazetteer point must sit to the subgraph to count as on it


def hops_from_cameras(graph, camera_indices: List[int]) -> np.ndarray:
    """Hop count from the nearest camera to every node (-1 where unreachable)."""
    n = len(graph.node_ids)
    src = graph.edge_index[0].numpy()
    dst = graph.edge_index[1].numpy()
    neighbours: List[List[int]] = [[] for _ in range(n)]
    for u, v in zip(src.tolist(), dst.tolist()):
        # Undirected for reach: a report one block away is as relevant whichever way the
        # traffic happens to flow down that block.
        neighbours[u].append(v)
        neighbours[v].append(u)

    hops = np.full(n, -1, dtype=np.int64)
    queue: deque = deque()
    for index in camera_indices:
        hops[index] = 0
        queue.append(index)
    while queue:
        u = queue.popleft()
        for v in neighbours[u]:
            if hops[v] == -1:
                hops[v] = hops[u] + 1
                queue.append(v)
    return hops


def load_camera_ids(path: Path = CAMERA_MAP_CSV) -> Dict[str, List[str]]:
    """Intersection label -> the camera ids watching it (several units can share a junction)."""
    out: Dict[str, List[str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["intersection"].strip(), []).append(row["camera_id"].strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-hops", type=int, default=DEFAULT_K,
                        help="furthest a report may be from a camera in road hops")
    parser.add_argument("--max-km", type=float, default=3.0,
                        help="and in straight-line km. Both limits apply: this subgraph is built "
                             "as k hops around the cameras, so a hop limit alone admits the whole "
                             "of it -- EDSA-Balintawak lands 8 hops from EDSA-Ortigas-Shaw and "
                             "9 km away, where it says nothing about what that camera sees")
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    graph = build_subgraph(args.spatial_dir, args.k)
    camera_indices = dict(graph.camera_nodes)
    print(f"k={args.k} subgraph: {len(graph.node_ids)} nodes, {len(camera_indices)} cameras")

    coords = pd.read_csv(args.spatial_dir / "full_network_static_features.csv",
                         usecols=["node_id", "lat", "lon"]).set_index("node_id").reindex(graph.node_ids)
    lat, lon = np.radians(coords["lat"].to_numpy()), np.radians(coords["lon"].to_numpy())

    def snap(point):
        la, lo = np.radians(point[0]), np.radians(point[1])
        h = np.sin((lat - la) / 2) ** 2 + np.cos(la) * np.cos(lat) * np.sin((lon - lo) / 2) ** 2
        km = 2 * 6371.0 * np.arcsin(np.sqrt(h))
        best = int(np.argmin(km))
        return (best, float(km[best]))

    # Every intersection the gazetteer can locate: the camera ones are already graph indices,
    # the rest are geocoded points that have to be snapped onto the subgraph.
    places: Dict[str, int] = dict(camera_indices)
    off_graph = 0
    if INTERSECTIONS_CSV.exists():
        with INTERSECTIONS_CSV.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                label = row["intersection"].strip()
                if label in places:
                    continue
                node, km = snap((float(row["lat"]), float(row["lon"])))
                if km <= SNAP_KM:
                    places[label] = node
                else:
                    off_graph += 1
    print(f"gazetteer intersections on this subgraph: {len(places)} ({off_graph} too far, dropped)")

    # Per-camera reach, computed once rather than once per intersection.
    per_camera = {label: hops_from_cameras(graph, [index]) for label, index in camera_indices.items()}
    cameras_by_intersection = load_camera_ids()

    def km_between(a: int, b: int) -> float:
        h = (np.sin((lat[b] - lat[a]) / 2) ** 2
             + np.cos(lat[a]) * np.cos(lat[b]) * np.sin((lon[b] - lon[a]) / 2) ** 2)
        return float(2 * 6371.0 * np.arcsin(np.sqrt(h)))

    rows, unreachable, no_camera = [], 0, 0
    for label, node in sorted(places.items()):
        best, winners = args.max_hops + 1, []
        for camera_label, distance in per_camera.items():
            d = int(distance[node])
            if d < 0 or d > args.max_hops:
                continue
            if km_between(node, camera_indices[camera_label]) > args.max_km:
                continue
            if d < best:
                best, winners = d, [camera_label]
            elif d == best:
                winners.append(camera_label)
        if not winners:
            unreachable += 1
            continue
        # A camera intersection with no working camera id (e.g. a faulty unit left out of
        # camera_nodes.csv) cannot receive text; its reports go to the next camera instead.
        targets = [(w, c) for w in winners for c in cameras_by_intersection.get(w, [])]
        if not targets:
            no_camera += 1
            continue
        for camera_label, camera_id in targets:
            rows.append({"intersection": label, "camera_intersection": camera_label,
                         "camera_id": camera_id, "hops": best,
                         "km": round(km_between(node, camera_indices[camera_label]), 2)})

    placed_labels = {r["intersection"] for r in rows}
    print(f"\nplaces that reach a camera within {args.max_hops} hops: {len(placed_labels)} of {len(places)}")
    print(f"  unreachable: {unreachable}   nearest camera has no working unit: {no_camera}")
    by_hop = pd.Series([r["hops"] for r in rows]).value_counts().sort_index()
    print(f"  hop distribution: {by_hop.to_dict()}")

    if args.dry_run:
        seen = set()
        for r in rows:
            if r["intersection"] in seen:
                continue          # one line per place, not per camera unit
            seen.add(r["intersection"])
            print(f"    {r['intersection']:34s} -> {r['camera_intersection']:26s} "
                  f"{r['hops']:2d} hops  {r['km']:5.2f} km")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["intersection", "camera_intersection", "camera_id", "hops", "km"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows ({len(placed_labels)} intersections) -> {args.out}")


if __name__ == "__main__":
    main()
