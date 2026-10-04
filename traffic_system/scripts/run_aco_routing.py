import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.graph_data import DEFAULT_K, build_subgraph  # noqa: E402
from src.routing.aco_routing import AntColonyConfig, RouteResult, ant_colony_shortest_path, multi_stop_route  # noqa: E402
from src.routing.departure_window import resolve_departure_window  # noqa: E402
from src.routing.dynamic_weight import DEFAULT_LAMBDA, DEFAULT_MISSING_RISK, weighted_graph_from_risk_file  # noqa: E402
from src.routing.eta_engine import DEFAULT_GAMMA, load_free_flow_seconds, path_eta_seconds  # noqa: E402

DEFAULT_CRITICAL_RISK = 0.6


def known_places(graph) -> Dict[str, int]:
    return dict(sorted(graph.camera_nodes.items()))


def resolve_known_place(graph, wg, name: str) -> int:
    places = known_places(graph)
    if name not in places:
        choices = ", ".join(places)
        raise SystemExit(f"unknown place {name!r}. Valid choices are: {choices}")
    return int(wg.node_ids[places[name]])


def describe_path(graph, node_ids: List[int]) -> str:
    reverse = {int(graph.node_ids[i]): label for label, i in graph.camera_nodes.items()}
    return " -> ".join((reverse.get(n, str(n)) for n in node_ids))


def label_for_node(graph, node_id: int) -> str:
    reverse = {int(graph.node_ids[i]): label for label, i in graph.camera_nodes.items()}
    return reverse.get(node_id, str(node_id))


def segment_breakdown(graph, wg, path: List[int], critical_risk: float) -> List[dict]:
    indices = [wg.index_of(n) for n in path]
    edges = [wg.edge_id(u, v) for u, v in zip(indices, indices[1:])]
    segments = []
    for edge, u, v in zip(edges, path, path[1:]):
        risk = float(wg.risk[edge])
        segments.append(
            {
                "from": label_for_node(graph, u),
                "to": label_for_node(graph, v),
                "from_node_id": u,
                "to_node_id": v,
                "distance_m": round(float(wg.distance[edge]), 2),
                "risk": round(risk, 4),
                "critical": risk >= critical_risk,
            }
        )
    return segments


def summarise(
    graph, wg, free_flow_seconds, gamma: float, critical_risk: float, departure_dt, result: RouteResult
) -> dict:

    def metrics_dict(m):
        eta_seconds = (
            path_eta_seconds(wg, free_flow_seconds, m.nodes, gamma) if free_flow_seconds is not None else None
        )
        segments = segment_breakdown(graph, wg, m.nodes, critical_risk)
        critical_points = [s for s in segments if s["critical"]]
        arrival_dt = (
            departure_dt + pd.Timedelta(seconds=eta_seconds)
            if eta_seconds is not None and departure_dt is not None
            else None
        )
        return {
            "path": describe_path(graph, m.nodes),
            "node_ids": m.nodes,
            "hops": m.n_edges,
            "distance_m": round(m.distance_m, 2),
            "risk_exposure": round(m.risk_exposure, 4),
            "mean_risk": round(m.mean_risk, 4),
            "dynamic_cost": round(m.dynamic_cost, 2),
            "departure_time": (
                departure_dt.isoformat(timespec="seconds") if departure_dt is not None else None
            ),
            "arrival_time": arrival_dt.isoformat(timespec="seconds") if arrival_dt is not None else None,
            "duration_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
            "duration_minutes": round(eta_seconds / 60.0, 1) if eta_seconds is not None else None,
            "segments": segments,
            "critical_points": critical_points,
            "critical_point_count": len(critical_points),
        }

    return {"best": metrics_dict(result.best), "alternatives": [metrics_dict(m) for m in result.alternatives]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--risk-edges", required=True, type=Path)
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("-k", type=int, default=DEFAULT_K)
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    p.add_argument(
        "--depart-at",
        default=None,
        help="when you plan to leave, e.g. 2026-05-25T17:30:00 (defaults to right now if omitted)",
    )
    p.add_argument("--missing-risk", type=float, default=DEFAULT_MISSING_RISK)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="ETA risk sensitivity")
    p.add_argument(
        "--critical-risk",
        type=float,
        default=DEFAULT_CRITICAL_RISK,
        help="a route segment at or above this risk is flagged as a critical point",
    )
    p.add_argument(
        "--no-eta",
        action="store_true",
        help="skip ETA (use if metro_manila_travel_time.npz has not been built yet)",
    )
    p.add_argument(
        "--list-places", action="store_true", help="print the valid origin/destination choices and exit"
    )
    p.add_argument("--origin")
    p.add_argument("--destination")
    p.add_argument("--stops", nargs="+", help="origin, one or more waypoints, destination, in order")
    p.add_argument("--n-alternatives", type=int, default=3)
    p.add_argument("--n-ants", type=int, default=20)
    p.add_argument("--n-iterations", type=int, default=60)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--evaporation", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/routes/route.json")
    a = p.parse_args()
    graph = build_subgraph(a.spatial_dir, a.k)
    if a.list_places:
        for name in known_places(graph):
            print(name)
        return
    if bool(a.origin) == bool(a.stops) or (a.origin and (not a.destination)):
        p.error("give either --origin and --destination, or --stops (not both)")
    depart_at = a.depart_at if a.depart_at is not None else datetime.now().isoformat()
    departure_dt = pd.to_datetime(depart_at)
    window = resolve_departure_window(a.risk_edges, depart_at)
    wg = weighted_graph_from_risk_file(
        graph, a.risk_edges, window=window, lam=a.lam, missing_risk=a.missing_risk
    )
    if wg.coverage < 1.0:
        print(
            f"WARNING: only {wg.coverage:.1%} of edges had a predicted risk; the rest fell back to {a.missing_risk}."
        )
    config = AntColonyConfig(
        n_ants=a.n_ants,
        n_iterations=a.n_iterations,
        alpha=a.alpha,
        beta=a.beta,
        evaporation=a.evaporation,
        n_alternatives=a.n_alternatives,
        seed=a.seed,
    )
    if a.stops:
        stops = [resolve_known_place(graph, wg, s) for s in a.stops]
        result = multi_stop_route(wg, stops, config)
    else:
        origin = resolve_known_place(graph, wg, a.origin)
        destination = resolve_known_place(graph, wg, a.destination)
        result = ant_colony_shortest_path(wg, origin, destination, config)
    free_flow_seconds = None
    if not a.no_eta:
        travel_time_path = a.spatial_dir / "metro_manila_travel_time.npz"
        if not travel_time_path.exists():
            print(
                f"WARNING: {travel_time_path} does not exist; skipping ETA. Re-run spatial_topology.py to generate it, or pass --no-eta to silence this."
            )
        else:
            free_flow_seconds = load_free_flow_seconds(a.spatial_dir, graph)
    summary = summarise(graph, wg, free_flow_seconds, a.gamma, a.critical_risk, departure_dt, result)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()