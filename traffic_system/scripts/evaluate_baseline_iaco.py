"""
Compare ANTROUTE, the Improved ACO baseline (Cheng 2023) and shortest distance
on the same trips.

    python scripts/evaluate_baseline_iaco.py --risk-edges <risk_edges.csv> --auto-labels <auto_labels.csv> [...]

Inputs per method:
  - ANTROUTE: predicted risk, cost = dist * (1 + lambda * risk), planned once.
  - Improved ACO: observed traffic (congestion labels + vehicle counts),
    re-planned when new data arrives (eq. 9).
  - Shortest distance: road length only.

All routes are scored the same way: driven edge by edge against the observed
traffic at that time. Observations exist only at cameras; other edges use the
nearest camera. Uses the test split by default.

Outputs trips.csv, summary.json and summary.md in --out-dir.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.graph_data import DEFAULT_K, GraphData, build_subgraph, resolve_camera_map  # noqa: E402
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path  # noqa: E402
from src.routing.baseline_iaco import (  # noqa: E402
    IacoConfig,
    IacoGraph,
    TrafficSnapshots,
    comprehensive_cost,
    drive,
    fill_from_nearest_camera,
    iaco_dynamic_trip,
    spatial_shortest_path,
)
from src.routing.dynamic_weight import DEFAULT_LAMBDA, build_weighted_graph, edge_distances  # noqa: E402
from src.routing.eta_engine import DEFAULT_GAMMA, congested_eta, load_free_flow_seconds  # noqa: E402

LOCAL_TZ = "Asia/Manila"
METHODS = ("antroute", "improved_aco", "shortest_distance")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def local_naive(values: pd.Series) -> pd.Series:
    """Parse timestamps as Manila local time without timezone."""
    ts = pd.to_datetime(values, format="ISO8601")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return ts


def iaco_graph(graph: GraphData, spatial_dir: Path) -> IacoGraph:
    coords = pd.read_csv(spatial_dir / "full_network_static_features.csv", usecols=["node_id", "lat", "lon"])
    coords = coords.set_index("node_id").reindex(graph.node_ids)
    if coords.isna().any().any():
        raise ValueError("some subgraph nodes have no coordinates in full_network_static_features.csv")
    return IacoGraph(
        node_ids=np.asarray(graph.node_ids),
        src=graph.edge_index[0].numpy(),
        dst=graph.edge_index[1].numpy(),
        distance=edge_distances(graph),
        lat=coords["lat"].to_numpy(dtype=float),
        lon=coords["lon"].to_numpy(dtype=float),
    )


def load_risk(path: Path, graph: GraphData, split: str) -> pd.DataFrame:
    cols = ["day", "session", "window_start", "window_end", "source_node_id", "target_node_id",
            "risk", "camera_edge", "split", "weak_target"]
    frame = pd.read_csv(path, usecols=cols)
    frame = frame[frame["split"] == split].copy()
    if frame.empty:
        raise SystemExit(f"no rows in split {split!r} of {path}")
    edge_of = {
        (int(u), int(v)): e
        for e, (u, v) in enumerate(zip(graph.node_ids[graph.edge_index[0].numpy()],
                                       graph.node_ids[graph.edge_index[1].numpy()]))
    }
    frame["edge"] = [edge_of.get((int(u), int(v)), -1) for u, v in zip(frame["source_node_id"], frame["target_node_id"])]
    frame = frame[frame["edge"] >= 0]
    frame["start"] = local_naive(frame["window_start"])
    frame["end"] = local_naive(frame["window_end"])
    return frame


def camera_labels_by_window(frame: pd.DataFrame, graph: GraphData) -> Dict[pd.Timestamp, Dict[int, float]]:
    """Observed congestion label per camera per window, taken from weak_target."""
    cameras = set(graph.camera_nodes.values())
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    owner = np.full(len(src), -1)
    for e, (u, v) in enumerate(zip(src, dst)):
        if (u in cameras) != (v in cameras):
            owner[e] = u if u in cameras else v
    labelled = frame[frame["weak_target"].notna()].copy()
    labelled["camera"] = owner[labelled["edge"].to_numpy()]
    labelled = labelled[labelled["camera"] >= 0]
    out: Dict[pd.Timestamp, Dict[int, float]] = {}
    for (end, camera), value in labelled.groupby(["end", "camera"])["weak_target"].mean().items():
        out.setdefault(end, {})[int(camera)] = float(value)
    return out


def load_vehicle_counts(paths: List[Path], graph: GraphData, camera_csv: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, usecols=["camera_id", "timestamp", "n_vehicles"]) for p in paths]
    counts = pd.concat(frames, ignore_index=True)
    mapping, unmapped = resolve_camera_map(sorted(counts["camera_id"].unique()), list(graph.camera_nodes), camera_csv)
    if unmapped:
        print(f"WARNING: {len(unmapped)} camera id(s) not mapped to an intersection, ignored: {unmapped[:5]}")
    counts["camera"] = counts["camera_id"].map(lambda c: graph.camera_nodes.get(mapping.get(c), -1))
    counts = counts[counts["camera"] >= 0].copy()
    counts["time"] = local_naive(counts["timestamp"])
    return counts.sort_values("time")


def flow_by_window(counts: pd.DataFrame, windows: pd.DataFrame) -> Dict[pd.Timestamp, Dict[int, float]]:
    """Mean vehicle count per camera for each window."""
    out: Dict[pd.Timestamp, Dict[int, float]] = {}
    by_camera = {c: (g["time"].to_numpy(), g["n_vehicles"].to_numpy(dtype=float)) for c, g in counts.groupby("camera")}
    for start, end in windows[["start", "end"]].itertuples(index=False):
        for camera, (times, values) in by_camera.items():
            lo, hi = np.searchsorted(times, np.datetime64(start)), np.searchsorted(times, np.datetime64(end))
            if hi > lo:
                out.setdefault(end, {})[int(camera)] = float(values[lo:hi].mean())
    return out


# ---------------------------------------------------------------------------
# per session: observed snapshots and predicted risk
# ---------------------------------------------------------------------------
def seconds(ts: pd.Timestamp) -> float:
    return float((ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1))


def build_session(
    g: IacoGraph,
    frame: pd.DataFrame,
    labels: Dict[pd.Timestamp, Dict[int, float]],
    flows: Dict[pd.Timestamp, Dict[int, float]],
    free_flow: np.ndarray,
    gamma: float,
) -> Tuple[Optional[TrafficSnapshots], List[pd.Timestamp], Dict[pd.Timestamp, np.ndarray]]:
    usable = sorted(t for t in frame["end"].unique() if labels.get(t) and flows.get(t))
    if not usable:
        return None, [], {}
    congestion = np.stack([fill_from_nearest_camera(g.src, g.dst, g.n_nodes, labels[t]) for t in usable])
    flow = np.stack([fill_from_nearest_camera(g.src, g.dst, g.n_nodes, flows[t]) for t in usable])
    travel = np.stack([congested_eta(free_flow, c, gamma) for c in congestion])
    snapshots = TrafficSnapshots(
        times=np.array([seconds(t) for t in usable]), travel_time=travel, flow=flow, congestion=congestion
    )
    predicted = {}
    for t, rows in frame[frame["end"].isin(usable)].groupby("end"):
        risk = np.zeros(g.n_edges)
        risk[rows["edge"].to_numpy()] = rows["risk"].to_numpy(dtype=float)
        predicted[t] = risk
    return snapshots, usable, predicted


def departures(times: List[pd.Timestamp], every_min: float) -> List[pd.Timestamp]:
    chosen: List[pd.Timestamp] = []
    for t in times:
        if not chosen or (t - chosen[-1]) >= pd.Timedelta(minutes=every_min):
            chosen.append(t)
    return chosen


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def score(g: IacoGraph, nodes: List[int], departure: float, traffic: TrafficSnapshots, s_dep: np.ndarray) -> dict:
    edges = g.edges_of([g.index_of(n) for n in nodes])
    travel, exposure = drive(g, edges, departure, traffic)
    return {
        "hops": len(edges),
        "length_m": float(g.distance[edges].sum()),
        "combined_cost": float(s_dep[edges].sum()),
        "travel_time_s": travel,
        "congestion_exposure": exposure,
    }


def summarise(trips: pd.DataFrame) -> dict:
    ok = trips[trips["success"]]
    complete = ok.groupby("trip")["method"].nunique()
    paired = ok[ok["trip"].isin(complete[complete == len(METHODS)].index)]
    summary = {
        "trips": int(trips["trip"].nunique()),
        "paired_trips": int(paired["trip"].nunique()),
        "success": {m: int(ok[ok["method"] == m].shape[0]) for m in METHODS},
        "mean_over_paired_trips": {},
    }
    for m in METHODS:
        rows = paired[paired["method"] == m]
        summary["mean_over_paired_trips"][m] = {
            c: round(float(rows[c].mean()), 4)
            for c in ("length_m", "combined_cost", "travel_time_s", "congestion_exposure", "converged_at",
                      "stable_from", "replans")
            if rows[c].notna().any()
        }
    if not paired.empty:
        wide = paired.pivot(index="trip", columns="method", values="travel_time_s")
        diff = wide["antroute"] - wide["improved_aco"]
        summary["antroute_vs_improved_aco_travel_time"] = {
            "mean_difference_s": round(float(diff.mean()), 2),
            "antroute_faster": int((diff < -1e-6).sum()),
            "same": int((diff.abs() <= 1e-6).sum()),
            "improved_aco_faster": int((diff > 1e-6).sum()),
        }
    return summary


def markdown_table(summary: dict) -> str:
    names = {"antroute": "ANTROUTE", "improved_aco": "Improved ACO (Cheng 2023)", "shortest_distance": "Spatial shortest distance"}
    lines = [
        f"Paired trips: {summary['paired_trips']} of {summary['trips']}",
        "",
        "| Method | Length (m) | Combined cost s | Realised travel time (s) | Congestion exposure | Iterations | Re-plans |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        r = summary["mean_over_paired_trips"].get(m, {})
        cell = lambda k: f"{r[k]:.2f}" if k in r else "n/a"  # noqa: E731
        lines.append(
            f"| {names[m]} | {cell('length_m')} | {cell('combined_cost')} | {cell('travel_time_s')} | "
            f"{cell('congestion_exposure')} | {cell('stable_from')} | {cell('replans')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--risk-edges", required=True, type=Path)
    p.add_argument("--auto-labels", required=True, type=Path, nargs="+")
    p.add_argument("--camera-map", type=Path, default=REPO_ROOT / "configs/camera_nodes.csv")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("-k", type=int, default=DEFAULT_K)
    p.add_argument("--split", default="test")
    p.add_argument("--departure-every", type=float, default=30.0, help="minutes between departures in a session")
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="travel-time sensitivity to congestion")
    p.add_argument("--n-ants", type=int, default=20)
    p.add_argument("--n-iterations", type=int, default=60)
    p.add_argument("--invert-eq9", action="store_true", help="sensitivity variant: tau * t_old / t_new")
    p.add_argument("--reset-on-replan", action="store_true", help="sensitivity variant: fresh pheromone per re-plan")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/baseline_comparison")
    a = p.parse_args()

    graph = build_subgraph(a.spatial_dir, a.k)
    g = iaco_graph(graph, a.spatial_dir)
    free_flow = load_free_flow_seconds(a.spatial_dir, graph)
    frame = load_risk(a.risk_edges, graph, a.split)
    labels = camera_labels_by_window(frame, graph)
    windows = frame[["start", "end"]].drop_duplicates()
    flows = flow_by_window(load_vehicle_counts(a.auto_labels, graph, a.camera_map), windows)

    aco_config = AntColonyConfig(n_ants=a.n_ants, n_iterations=a.n_iterations, seed=a.seed)
    iaco_config = IacoConfig(n_ants=a.n_ants, n_iterations=a.n_iterations, seed=a.seed,
                             invert_eq9=a.invert_eq9, reset_on_replan=a.reset_on_replan)
    camera_ids = {label: int(graph.node_ids[i]) for label, i in graph.camera_nodes.items()}
    pairs = [(o, d) for o in camera_ids for d in camera_ids if o != d]

    rows: List[dict] = []
    trip = 0
    for (day, session), session_frame in frame.groupby(["day", "session"]):
        traffic, usable, predicted = build_session(g, session_frame, labels, flows, free_flow, a.gamma)
        if traffic is None:
            print(f"{day} {session}: no window with both a label and a vehicle count, skipped")
            continue
        starts = departures(usable, a.departure_every)
        print(f"{day} {session}: {len(usable)} observed windows, {len(starts)} departures", flush=True)
        for start in starts:
            dep = seconds(start)
            w = traffic.index_at(dep)
            s_dep = comprehensive_cost(g.distance, traffic.travel_time[w], traffic.flow[w])
            wg = build_weighted_graph(graph, predicted[start], lam=a.lam)
            print(f"  departure {start:%H:%M}", flush=True)
            for o_label, d_label in pairs:
                origin, destination = camera_ids[o_label], camera_ids[d_label]
                try:
                    shortest = spatial_shortest_path(g, origin, destination)
                except ValueError:
                    continue   # not connected inside the k-hop subgraph
                trip += 1
                base = {"trip": trip, "day": day, "session": session, "departure": start.isoformat(),
                        "origin": o_label, "destination": d_label}

                rows.append({**base, "method": "shortest_distance", "success": True,
                             **score(g, shortest, dep, traffic, s_dep)})
                try:
                    best = ant_colony_shortest_path(wg, origin, destination, aco_config).best
                    rows.append({**base, "method": "antroute", "success": True,
                                 **score(g, best.nodes, dep, traffic, s_dep)})
                except ValueError as err:
                    rows.append({**base, "method": "antroute", "success": False, "error": str(err)})
                try:
                    dyn = iaco_dynamic_trip(g, origin, destination, dep, traffic, iaco_config)
                    rows.append({**base, "method": "improved_aco", "success": True,
                                 **score(g, dyn.path, dep, traffic, s_dep),
                                 "converged_at": dyn.converged_at, "stable_from": dyn.stable_from,
                                 "replans": dyn.replans,
                                 "initial_combined_cost": dyn.initial_cost})
                except ValueError as err:
                    rows.append({**base, "method": "improved_aco", "success": False, "error": str(err)})

    if not rows:
        raise SystemExit("no trips could be evaluated; check that the labels, vehicle counts and split overlap")
    trips = pd.DataFrame(rows)
    for column in ("converged_at", "stable_from", "replans"):
        if column not in trips:
            trips[column] = np.nan
    summary = summarise(trips)
    summary["settings"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()
                           if k not in ("auto_labels",)}
    summary["settings"]["auto_labels"] = [str(x) for x in a.auto_labels]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    trips.to_csv(a.out_dir / "trips.csv", index=False)
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (a.out_dir / "summary.md").write_text(markdown_table(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\n" + markdown_table(summary))
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
