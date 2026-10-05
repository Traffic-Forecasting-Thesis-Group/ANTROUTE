"""
Wires the real CNN+LSTM -> RADR STGNN -> MLP Decoder -> Dynamic Weight Engine -> ACO
pipeline into the FastAPI route-planning endpoint, in place of route_engine.py's
haversine-distance placeholder.

No live data feed exists (CCTV/Twitter/weather are all historical), so this always
routes against the most recent scored window in risk_edges.csv -- the newest decoder
run stands in for "right now". Picking a specific recorded day/time instead is a
--window flag away in the underlying scripts for anyone building that into a demo.

Routing covers the whole Metro Manila road network (build_full_graph), not just the
k-hop subgraph the STGNN trains on -- that subgraph exists only because the model
needs a dense adjacency, a constraint routing does not share. Congestion-risk
awareness still only applies where a camera informs it; elsewhere an edge is assumed
typical (see missing_risk below) rather than risk-free or risk-laden. An origin or
destination further than MAX_SNAP_KM from every road in the network raises
RouteOutsideNetworkError rather than silently inventing a route; route_engine.
plan_routes() catches this and falls back to the distance-only placeholder.
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
import pandas as pd

LOCAL_TZ = "Asia/Manila"

from app.config import settings
from src.data.graph_data import GraphData, build_full_graph
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path, multi_stop_route
from src.routing.baseline_iaco import IacoConfig, IacoGraph
from src.routing.baseline_router import (
    IACO,
    SHORTEST_DISTANCE,
    BaselineInputs,
    build_inputs,
    camera_observations,
    plan_baseline,
)
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

# The baseline colony: Cheng's parameters (alpha 1, beta 5, rho 0.3, q0 0.7 for a large
# network) at baseline_iaco's 20 ants x 60 iterations. The paper's own 1.5 ants per node
# would be ~89,000 ants per iteration on this graph. This is more search than ANTROUTE gets
# below (8 x 15), which errs in the baseline's favour.
BASELINE_CONFIG = IacoConfig(seed=0)


class RouteOutsideNetworkError(Exception):
    """Origin or destination is too far from any node in the monitored subgraph."""


@dataclass
class _Network:
    graph: GraphData
    wg_antroute: WeightedGraph  # lambda = DEFAULT_LAMBDA, the real risk-aware engine
    risk_edges_path: Path       # the scored file, re-read once for the baseline's observations
    free_flow_seconds: Optional[np.ndarray]
    node_coords: Dict[int, Tuple[float, float]]  # node_id -> (lat, lon)
    risk_low: float  # tercile cutoffs over this window's risk, for labelling
    risk_high: float  # routes clear / moderate / heavy
    events: List[TrafficEvent]  # parsed MMDA incidents, placed on the graph
    event_intersections: Dict[str, int]  # incident label -> graph index (camera_nodes + non-camera landmarks)
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

    # Where an incident label places on the graph. Starts from the 8 camera
    # intersections (already graph indices) and adds the non-camera EDSA landmarks
    # the MMDA feed also names often -- Guadalupe, Magallanes, Balintawak, etc. --
    # geocoded once and snapped to the nearest real node (configs/event_intersections.csv
    # records the source coordinates and the snap, so a bad match is auditable).
    event_intersections: Dict[str, int] = dict(graph.camera_nodes)
    node_index = {int(n): i for i, n in enumerate(graph.node_ids)}
    extra_path = REPO_ROOT / "configs/event_intersections.csv"
    if extra_path.exists():
        with extra_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                label = row["intersection"].strip()
                if label in event_intersections:
                    continue
                point = (float(row["lat"]), float(row["lon"]))
                try:
                    node_id = _nearest_node_id(node_coords, point)
                except RouteOutsideNetworkError:
                    continue
                event_intersections[label] = node_index[node_id]

    return _Network(
        graph, wg_antroute, risk_edges_path, free_flow_seconds, node_coords, float(risk_low), float(risk_high),
        events, event_intersections,
    )


def _observed_rows(path: Path, window: Optional[str]) -> pd.DataFrame:
    """
    The camera observations (weak_target) recorded for one window, read with only the four
    columns they need -- the full file is ~100MB and _network() has already parsed it.
    """
    empty = pd.DataFrame(columns=["source_node_id", "target_node_id", "weak_target"])
    header = pd.read_csv(path, nrows=0).columns
    if window is None or "weak_target" not in header:
        return empty
    key = "window_start" if "window_start" in header else "window_end"
    frame = pd.read_csv(path, usecols=[key, "source_node_id", "target_node_id", "weak_target"])
    return frame[frame[key].astype(str) == window]


@lru_cache(maxsize=1)
def _baseline_inputs() -> BaselineInputs:
    """
    What the baseline plans with: Cheng's cost on the same city graph ANTROUTE routes on, at
    the same window, from what the cameras observed then (see src/routing/baseline_router.py).
    Built once, like _network().
    """
    net = _network()
    wg = net.wg_antroute
    g = IacoGraph(
        node_ids=np.asarray(net.graph.node_ids),
        src=wg.src,
        dst=wg.dst,
        distance=wg.distance,
        lat=np.array([net.node_coords[int(n)][0] for n in net.graph.node_ids]),
        lon=np.array([net.node_coords[int(n)][1] for n in net.graph.node_ids]),
    )
    free_flow = (
        net.free_flow_seconds
        if net.free_flow_seconds is not None
        else wg.distance / (FALLBACK_SPEED_KMH / 3.6)
    )
    observed = camera_observations(
        _observed_rows(net.risk_edges_path, wg.window),
        np.asarray(net.graph.node_ids),
        list(net.graph.camera_nodes.values()),
    )
    # No vehicle counts ship with the app (they live with the YOLO detections), so the flow
    # term of Cheng's cost is zero here; build_inputs records that in flow_observed.
    return build_inputs(g, free_flow, observed, camera_flow=None)


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


def _nearest_node_id(node_coords: Dict[int, Tuple[float, float]], point: Tuple[float, float]) -> int:
    """
    Closest node id to (lat, lng) by plain dict scan; raises if nothing is within
    MAX_SNAP_KM. Used while building the network, before the vectorised coordinate
    arrays nearest_node() uses exist -- not the hot path, so the O(n) scan is fine for
    the handful of landmarks configs/event_intersections.csv snaps at startup.
    """
    best_id, best_km = None, float("inf")
    for node_id, coord in node_coords.items():
        km = _haversine_km(point, coord)
        if km < best_km:
            best_id, best_km = node_id, km
    if best_km > MAX_SNAP_KM:
        raise RouteOutsideNetworkError(
            f"({point[0]:.4f}, {point[1]:.4f}) is {best_km:.1f} km from the nearest road in the "
            f"network -- outside the routable area"
        )
    return best_id


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


def _duration_min(net: _Network, m: PathMetrics) -> int:
    """
    Travel time of a route by the app's one ETA engine, whichever system chose the route.

    Both systems' cards are timed with this, so two systems that pick the same road always
    show the same time, and a difference on the map is a difference in the route. Each
    system's own ETA prediction is a separate question, scored offline against Apple Maps
    by scripts/evaluate_routing.py.
    """
    if net.free_flow_seconds is None:
        return max(1, round(m.distance_m / 1000 / FALLBACK_SPEED_KMH * 60))
    return max(1, round(path_eta_seconds(net.wg_antroute, net.free_flow_seconds, m.nodes) / 60))


def _metrics_to_option(net: _Network, wg: WeightedGraph, label: str, m: PathMetrics) -> dict:
    return {
        "label": label,
        "via": f"Risk-aware route (ACO) -- {m.n_edges} segments, {m.mean_risk:.0%} avg. risk",
        "duration_min": _duration_min(net, m),
        "distance_km": round(m.distance_m / 1000, 2),
        "congestion_level": _congestion_label(net, m.mean_risk),
        "mean_risk": round(m.mean_risk, 4),
        "algorithm": "antroute",
        "fallback_reason": None,
        "event_note": None,
        "path": _path_points(net, m.nodes),
    }


def _baseline_option(net: _Network, stops: List[int]) -> dict:
    """
    The baseline's single route (IACO returns one optimal path), in the same dict shape as
    ANTROUTE's options.

    Distance, risk and duration are all measured the same way as ANTROUTE's (on its graph,
    by _duration_min), so the two cards compare routes, not measuring methods.
    """
    route = plan_baseline(_baseline_inputs(), stops, BASELINE_CONFIG, fallback=settings.baseline_fallback)
    m = net.wg_antroute.evaluate_path(route.nodes)
    if route.algorithm == IACO:
        method = "Improved ACO (Cheng 2023)"
    elif route.algorithm == SHORTEST_DISTANCE:
        method = "Spatial shortest distance (IACO found no route)"
    else:
        method = f"IACO for {route.legs_by_iaco} of {route.legs} legs, shortest distance for the rest"
    return {
        "label": "Baseline route",
        "via": f"{method} -- {m.n_edges} segments, {m.mean_risk:.0%} avg. risk",
        "duration_min": _duration_min(net, m),
        "distance_km": round(m.distance_m / 1000, 2),
        "congestion_level": _congestion_label(net, m.mean_risk),
        "mean_risk": round(m.mean_risk, 4),
        "algorithm": route.algorithm,
        "fallback_reason": route.fallback_reason,
        "event_note": None,
        "path": _path_points(net, route.nodes),
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
    Routes for the given stops, in the same dict shape route_engine.plan_routes() returns.

    model="antroute" is the risk- and event-aware ACO, with up to three options.
    model="baseline" is the thesis baseline, Improved ACO (Cheng 2023), with one option.
    See _baseline_option() and src/routing/baseline_router.py.

    Raises RouteOutsideNetworkError if any stop is too far from the road network, and
    BaselineNoRouteError when the baseline finds nothing and its fallback is switched off.

    For 3+ stops (one or more waypoints), the underlying multi_stop_route() does not
    produce alternatives (it chains the single best leg-by-leg route), so only one
    option is returned rather than three.
    """
    net = _network()
    stops = [nearest_node(net, origin_coords)] + [nearest_node(net, c) for c in destination_coords]

    # The baseline plans on observed traffic with Cheng's own cost and colony. It sees neither
    # ANTROUTE's predicted risk nor the reported incidents: those are what ANTROUTE adds.
    if model == "baseline":
        return [_baseline_option(net, stops)]

    wg = net.wg_antroute
    impact = None
    live = live_events(net)
    if live:
        impact = event_impact(net.graph, live, net.event_intersections)
        wg = apply_events(wg, impact, mu=settings.event_mu)

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
