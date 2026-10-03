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
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

LOCAL_TZ = "Asia/Manila"

from app.config import settings
from src.data.graph_data import GraphData, build_full_graph
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path, multi_stop_route
from src.routing.dynamic_weight import DEFAULT_LAMBDA, PathMetrics, WeightedGraph, weighted_graph_from_risk_file
from src.routing.eta_engine import load_free_flow_seconds, path_eta_seconds
from src.routing.event_layer import (
    DEFAULT_MU,
    TrafficEvent,
    active_at,
    apply_events,
    event_impact,
    load_landmarks,
    parse_alerts,
)

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
    events: List[TrafficEvent]  # parsed MMDA incidents, placed on the graph
    coord_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    coord_lat: np.ndarray = field(default_factory=lambda: np.empty(0))
    coord_lon: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        # Snapping a searched place scans every node, and the routable graph is now the
        # whole city (59,521 nodes), so keep the coordinates as arrays and let numpy do
        # it rather than looping in Python on every request.
        if self.coord_ids.size == 0 and self.node_coords:
            items = list(self.node_coords.items())
            self.coord_ids = np.fromiter((n for n, _ in items), dtype=np.int64, count=len(items))
            self.coord_lat = np.fromiter((c[0] for _, c in items), dtype=np.float64, count=len(items))
            self.coord_lon = np.fromiter((c[1] for _, c in items), dtype=np.float64, count=len(items))


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
        raise ValueError(f"{len(missing)} graph node(s) missing from full_network_static_features.csv")
    return coords


@lru_cache(maxsize=1)
def _network() -> _Network:
    # The whole city, not the k-hop training subgraph: the subgraph exists because the
    # STGNN needs a dense adjacency, a constraint routing does not share. Risk is still
    # only known on the camera-covered edges -- the rest fall back to `missing_risk`.
    graph = build_full_graph(SPATIAL_DIR)
    risk_edges_path = Path(settings.risk_edges_path)
    if not risk_edges_path.is_absolute():
        risk_edges_path = REPO_ROOT / risk_edges_path
    if not risk_edges_path.exists():
        raise FileNotFoundError(risk_edges_path)

    # What to assume for an edge no camera informs. 0 would be actively wrong here: a
    # scored edge at the observed mean risk costs ~2.1x its length at lambda=2, so an
    # unscored edge assumed risk-free would look less than half the price, and the
    # router would systematically abandon the monitored corridor for roads it knows
    # nothing about. Assuming the observed mean instead makes "unknown" cost the same
    # as "typical", so coverage gaps neither attract nor repel a route.
    scored = weighted_graph_from_risk_file(graph, risk_edges_path, lam=DEFAULT_LAMBDA, missing_risk=0.0)
    observed = scored.risk[scored.risk > 0]
    missing_risk = float(observed.mean()) if observed.size else 0.5

    wg_antroute = weighted_graph_from_risk_file(
        graph, risk_edges_path, lam=DEFAULT_LAMBDA, missing_risk=missing_risk
    )
    wg_baseline = wg_antroute.with_lambda(0.0)

    free_flow_seconds = None
    if (SPATIAL_DIR / "metro_manila_travel_time.npz").exists():
        free_flow_seconds = load_free_flow_seconds(SPATIAL_DIR, graph)

    node_coords = _load_node_coords(SPATIAL_DIR, graph.node_ids)
    # Terciles over the edges a camera actually informs. Taken over every edge they
    # would mostly measure the imputed constant, and every route would come back the
    # same colour.
    risk_low, risk_high = np.percentile(observed if observed.size else wg_antroute.risk, [33, 66])

    # Event layer. Parsing the whole tweet corpus takes a few seconds, so it happens
    # once here rather than per request. An absent corpus is not fatal -- the system
    # just falls back to risk-only routing.
    events: List[TrafficEvent] = []
    raw_twitter = REPO_ROOT / "data/raw/twitter"
    landmarks = load_landmarks(REPO_ROOT / "configs/event_landmarks.csv")
    if raw_twitter.exists() and landmarks:
        events = parse_alerts(raw_twitter, landmarks)

    return _Network(
        graph, wg_antroute, wg_baseline, free_flow_seconds, node_coords, float(risk_low), float(risk_high), events
    )


def _as_of(net: _Network) -> Optional[datetime]:
    """
    The moment the system is routing "as of".

    There is no live feed, so this is the timestamp of the risk snapshot being served
    (the window predict_congestion_risk.py scored), which keeps the predicted-congestion
    and incident signals on the same clock. EVENT_AS_OF overrides it, which is what a
    demo uses to sit at a recorded moment that has incidents on the board.
    """
    override = (settings.event_as_of or "").strip()
    raw = override or (net.wg_antroute.window or "")
    if not raw:
        return None
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=ZoneInfo(LOCAL_TZ))


def live_events(net: _Network) -> List[tuple]:
    """(event, strength) pairs in effect at the moment this snapshot represents."""
    when = _as_of(net)
    if when is None or not net.events:
        return []
    return active_at(net.events, when)


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
    """Closest graph node id to (lat, lng); raises if nothing is within MAX_SNAP_KM."""
    lat, lon = math.radians(point[0]), math.radians(point[1])
    lat2, lon2 = np.radians(net.coord_lat), np.radians(net.coord_lon)
    h = np.sin((lat2 - lat) / 2) ** 2 + math.cos(lat) * np.cos(lat2) * np.sin((lon2 - lon) / 2) ** 2
    km = 2 * 6371.0 * np.arcsin(np.sqrt(h))

    best = int(np.argmin(km))
    if km[best] > MAX_SNAP_KM:
        raise RouteOutsideNetworkError(
            f"({point[0]:.4f}, {point[1]:.4f}) is {km[best]:.1f} km from the nearest road in the "
            f"network -- outside the routable area"
        )
    return int(net.coord_ids[best])


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


def _event_note(
    net: _Network,
    wg: WeightedGraph,
    impact: Optional[np.ndarray],
    live: Sequence[tuple],
    stops: Sequence[int],
    config: AntColonyConfig,
) -> Optional[str]:
    """
    A note naming the incidents that actually bear on *this* route, or None.

    Saying "routing around N incidents" on a trip nowhere near any of them would be
    false, and the incidents the router successfully avoids are by definition absent
    from the route it returns -- so neither the returned path nor the raw incident
    count answers the question on its own. What does: route the same trip with the
    event penalty switched off and see whether that path would have run into one.
    """
    if impact is None or not live or len(stops) != 2:
        return None
    try:
        without = ant_colony_shortest_path(net.wg_antroute, stops[0], stops[1], config).best
    except (ValueError, KeyError):
        return None

    indices = [net.wg_antroute.index_of(n) for n in without.nodes]
    edges = [net.wg_antroute.edge_id(u, v) for u, v in zip(indices, indices[1:])]
    hit = [e for e in edges if impact[e] > 0]
    if not hit:
        return None

    worst = max(live, key=lambda pair: pair[1])[0]
    return (
        f"{worst.kind.capitalize()} reported at {worst.location_text} "
        f"({worst.lanes_occupied} lane(s) occupied) - route adjusted to avoid it"
    )


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

    # Event-awareness is what ANTROUTE adds over the baseline: the baseline routes on
    # distance alone, with neither predicted congestion nor reported incidents, so it
    # stays the honest "no model at all" comparison.
    impact = None
    live: List[tuple] = []
    if model == "antroute":
        live = live_events(net)
        if live:
            impact = event_impact(net.graph, live)
            wg = apply_events(wg, impact, mu=settings.event_mu)

    stops = [nearest_node(net, origin_coords)] + [nearest_node(net, c) for c in destination_coords]

    # Lighter than the library default (20 ants x 60 iterations) because the colony now
    # searches the whole city. With the goal-directed heuristic the ants converge on the
    # optimum almost immediately -- measured on a 165-hop cross-city route, 8x15 returns
    # the identical path as 20x60 in 0.9s instead of 8.7s -- so the extra tours buy
    # response time, not route quality. They do still buy alternatives, which is why
    # this is not cut further.
    config = AntColonyConfig(n_ants=8, n_iterations=15, seed=0)
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
    note = _event_note(net, wg, impact, live, stops, config)
    if note:
        options[0]["event_note"] = note
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
