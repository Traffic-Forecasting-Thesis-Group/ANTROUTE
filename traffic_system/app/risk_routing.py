"""
Wires the real CNN+LSTM -> RADR STGNN -> MLP Decoder -> Dynamic Weight Engine -> ACO
pipeline into the FastAPI route-planning endpoint, in place of route_engine.py's
haversine-distance placeholder.

No live data feed exists (CCTV/Twitter/weather are all historical), so this always
routes against the most recent scored window in risk_edges.csv -- the newest decoder
run stands in for "right now". Picking a specific recorded day/time instead is a
--window flag away in the underlying scripts for anyone building that into a demo.

Routing is only possible between locations within the k-hop subgraph around the 8
CCTV intersections (the architecture's documented scope), so an origin or destination
far from every subgraph node raises RouteOutsideNetworkError rather than silently
inventing a route; route_engine.plan_routes() catches this and falls back to the
distance-only placeholder.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from src.data.graph_data import DEFAULT_K, GraphData, build_subgraph
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path, multi_stop_route
from src.routing.dynamic_weight import DEFAULT_LAMBDA, PathMetrics, WeightedGraph, weighted_graph_from_risk_file
from src.routing.eta_engine import load_free_flow_seconds, path_eta_seconds

REPO_ROOT = Path(__file__).resolve().parents[1]
SPATIAL_DIR = REPO_ROOT / "data/processed/spatial"
MAX_SNAP_KM = 2.0  # how far a searched place may be from the nearest subgraph node
FALLBACK_SPEED_KMH = 25.0  # used only if metro_manila_travel_time.npz is missing


class RouteOutsideNetworkError(Exception):
    """Origin or destination is too far from any node in the monitored subgraph."""


@dataclass
class _Network:
    graph: GraphData
    wg_antroute: WeightedGraph  # lambda = DEFAULT_LAMBDA, the real risk-aware engine
    wg_baseline: WeightedGraph  # lambda = 0, distance-only, same ACO search otherwise
    free_flow_seconds: Optional[np.ndarray]
    node_coords: Dict[int, Tuple[float, float]]  # node_id -> (lat, lon)
    risk_low: float  # tercile cutoffs over this window's risk, for labelling
    risk_high: float  # routes clear / moderate / heavy


def _load_node_coords(spatial_dir: Path, node_ids: np.ndarray) -> Dict[int, Tuple[float, float]]:
    wanted = {int(n) for n in node_ids}
    coords: Dict[int, Tuple[float, float]] = {}
    with (spatial_dir / "full_network_static_features.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            node_id = int(row["node_id"])
            if node_id in wanted:
                coords[node_id] = (float(row["lat"]), float(row["lon"]))
    missing = wanted - coords.keys()
    if missing:
        raise ValueError(f"{len(missing)} subgraph node(s) missing from full_network_static_features.csv")
    return coords


@lru_cache(maxsize=1)
def _network() -> _Network:
    graph = build_subgraph(SPATIAL_DIR, DEFAULT_K)
    risk_edges_path = Path(settings.risk_edges_path)
    if not risk_edges_path.is_absolute():
        risk_edges_path = REPO_ROOT / risk_edges_path
    if not risk_edges_path.exists():
        raise FileNotFoundError(risk_edges_path)
    wg_antroute = weighted_graph_from_risk_file(graph, risk_edges_path, lam=DEFAULT_LAMBDA)
    wg_baseline = wg_antroute.with_lambda(0.0)

    free_flow_seconds = None
    if (SPATIAL_DIR / "metro_manila_travel_time.npz").exists():
        free_flow_seconds = load_free_flow_seconds(SPATIAL_DIR, graph)

    node_coords = _load_node_coords(SPATIAL_DIR, graph.node_ids)
    risk_low, risk_high = np.percentile(wg_antroute.risk, [33, 66])

    return _Network(graph, wg_antroute, wg_baseline, free_flow_seconds, node_coords, float(risk_low), float(risk_high))


def available() -> bool:
    """False (rather than raising) when the scored risk file isn't present on this
    machine -- lets route_engine.py fall back to the placeholder instead of failing
    every request, e.g. on a teammate's machine without the Drive-synced CSV."""
    try:
        _network()
        return True
    except FileNotFoundError:
        return False


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def nearest_node(net: _Network, point: Tuple[float, float]) -> int:
    """Closest subgraph node id to (lat, lng); raises if nothing is within MAX_SNAP_KM."""
    best_node, best_km = None, float("inf")
    for node_id, coord in net.node_coords.items():
        km = _haversine_km(point, coord)
        if km < best_km:
            best_node, best_km = node_id, km
    if best_km > MAX_SNAP_KM:
        raise RouteOutsideNetworkError(
            f"({point[0]:.4f}, {point[1]:.4f}) is {best_km:.1f} km from the nearest monitored "
            f"intersection -- outside the routable network"
        )
    return best_node


def _congestion_label(net: _Network, mean_risk: float) -> str:
    if mean_risk <= net.risk_low:
        return "clear"
    if mean_risk <= net.risk_high:
        return "moderate"
    return "heavy"


def _path_points(net: _Network, node_ids: List[int]) -> List[dict]:
    return [{"lat": net.node_coords[n][0], "lng": net.node_coords[n][1]} for n in node_ids]


def _metrics_to_option(net: _Network, wg: WeightedGraph, label: str, m: PathMetrics) -> dict:
    eta_seconds = (
        path_eta_seconds(wg, net.free_flow_seconds, m.nodes) if net.free_flow_seconds is not None else None
    )
    duration_min = (
        max(1, round(eta_seconds / 60))
        if eta_seconds is not None
        else max(1, round(m.distance_m / 1000 / FALLBACK_SPEED_KMH * 60))
    )
    method = "Risk-aware route (ACO)" if wg.lam > 0 else "Shortest-distance baseline"
    return {
        "label": label,
        "via": f"{method} -- {m.n_edges} segments, {m.mean_risk:.0%} avg. risk",
        "duration_min": duration_min,
        "distance_km": round(m.distance_m / 1000, 2),
        "congestion_level": _congestion_label(net, m.mean_risk),
        "event_note": None,
        "path": _path_points(net, m.nodes),
    }


def plan_real_routes(
    origin_coords: Tuple[float, float],
    destination_coords: List[Tuple[float, float]],
    model: str = "antroute",
) -> List[dict]:
    """
    Real risk-aware (or distance-only baseline) routes for the given stops, in the
    same dict shape route_engine.plan_routes() returns.

    Raises RouteOutsideNetworkError if any stop is too far from the monitored k-hop
    subgraph -- the caller should fall back to the placeholder in that case, not
    fail the request outright.

    For 3+ stops (one or more waypoints), the underlying multi_stop_route() does not
    produce alternatives (it chains the single best leg-by-leg route), so only one
    option is returned rather than three.
    """
    net = _network()
    wg = net.wg_antroute if model == "antroute" else net.wg_baseline

    stops = [nearest_node(net, origin_coords)] + [nearest_node(net, c) for c in destination_coords]

    config = AntColonyConfig(seed=0)
    if len(stops) == 2:
        result = ant_colony_shortest_path(wg, stops[0], stops[1], config)
    else:
        result = multi_stop_route(wg, stops, config)

    candidates = [result.best] + result.alternatives
    by_risk = sorted(candidates, key=lambda m: m.risk_exposure)
    by_distance = sorted(candidates, key=lambda m: m.distance_m)

    options = [
        _metrics_to_option(net, wg, "Best route", result.best),
        _metrics_to_option(net, wg, "Least traffic", by_risk[0]),
        _metrics_to_option(net, wg, "Shortest distance", by_distance[0]),
    ]
    # De-duplicate: a 2-stop trip with no meaningfully different alternative (or any
    # multi-stop trip, which never has alternatives) would otherwise repeat one route
    # under all three labels.
    seen, deduped = set(), []
    for opt in options:
        key = tuple(p["lat"] for p in opt["path"])
        if key not in seen:
            seen.add(key)
            deduped.append(opt)
    return deduped
