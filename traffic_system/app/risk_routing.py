"""
Wires the real CNN+LSTM -> RADR STGNN -> MLP Decoder -> Dynamic Weight Engine -> ACO
pipeline into the FastAPI route-planning endpoint.

No live data feed exists (CCTV/Twitter/weather are all historical), so a trip is
routed against the recorded window in risk_edges.csv that matches its departure time
-- same time of day, preferring the same weekday (departure_window.
match_recorded_window). Only the AM and PM peaks were recorded; a departure outside
them uses the closest peak window and says so in the response.

risk_edges.csv is not in git (~100MB). Without it the routes are still real -- the
same road graph, ACO search and free-flow ETAs -- just with no congestion risk, and
the response says so rather than passing that off as traffic-aware.

Routing covers the whole Metro Manila road network (build_full_graph), not just the
k-hop subgraph the STGNN trains on -- that subgraph exists only because the model
needs a dense adjacency, a constraint routing does not share. Congestion-risk
awareness still only applies where a camera informs it; elsewhere an edge is assumed
typical (see missing_risk below) rather than risk-free or risk-laden. An origin or
destination further than MAX_SNAP_KM from every road in the network raises
RouteOutsideNetworkError rather than silently inventing a route.
"""

from __future__ import annotations

import csv
import logging
import math
import threading
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
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path, diverse_routes
from src.routing.departure_window import match_recorded_window
from src.routing.dynamic_weight import (
    DEFAULT_LAMBDA,
    RISK_KEY,
    PathMetrics,
    WeightedGraph,
    build_weighted_graph,
    load_risk_edges,
    risk_vector,
    window_key,
)
from src.routing.eta_engine import TYPICAL_CONGESTION_MULTIPLIER, load_free_flow_seconds, path_eta_seconds
from src.routing.event_layer import (
    DEFAULT_MU,
    TrafficEvent,
    active_at,
    apply_events,
    event_impact,
    load_landmarks,
    parse_alerts,
)

logger = logging.getLogger("uvicorn.error")

REPO_ROOT = Path(__file__).resolve().parents[1]
SPATIAL_DIR = REPO_ROOT / "data/processed/spatial"
MAX_SNAP_KM = 2.0  # how far a searched place may be from the nearest road node
FALLBACK_SPEED_KMH = 25.0  # used only if metro_manila_travel_time.npz is missing

NO_RISK_NOTE = "Congestion data is not loaded on the server - ETA assumes average Metro Manila traffic"


class RouteOutsideNetworkError(Exception):
    """Origin or destination is too far from any road in the network."""


class NoRouteError(Exception):
    """The stops are on the network but no drivable path joins them."""


@dataclass
class _Network:
    graph: GraphData
    risk_by_window: Dict[str, pd.DataFrame]  # window_start -> that window's decoder rows; empty without risk_edges.csv
    free_flow_seconds: Optional[np.ndarray]
    node_coords: Dict[int, Tuple[float, float]]  # node_id -> (lat, lon)
    risk_low: float  # tercile cutoffs over every window's risk, for labelling
    risk_high: float  # routes clear / moderate / heavy
    events: List[TrafficEvent]  # parsed MMDA incidents, placed on the graph
    event_intersections: Dict[str, int]  # incident label -> graph index (camera_nodes + non-camera landmarks)
    edge_names: Optional[np.ndarray] = None  # [E] OSM road name per edge id, None where unnamed
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


def _load_edge_names(spatial_dir: Path, graph: GraphData) -> Optional[np.ndarray]:
    """Road name per edge id from edge_names.csv (scripts/build_edge_names.py), or None
    if that file hasn't been built -- routes then just aren't described by road."""
    path = spatial_dir / "edge_names.csv"
    if not path.exists():
        logger.warning("%s not found: routes won't name their roads", path)
        return None
    with path.open(encoding="utf-8", newline="") as f:
        names = {(int(r["source_node_id"]), int(r["target_node_id"])): r["name"] for r in csv.DictReader(f)}
    node_ids = np.asarray(graph.node_ids)
    src, dst = node_ids[graph.edge_index[0].numpy()], node_ids[graph.edge_index[1].numpy()]
    return np.array([names.get((int(u), int(v))) for u, v in zip(src, dst)], dtype=object)


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


_network_lock = threading.Lock()


def _network() -> _Network:
    # Building takes ~40s (mostly parsing the tweet corpus), and main.py starts it in a
    # background thread at startup. The lock makes a request that arrives meanwhile
    # wait for that build instead of starting a second one.
    with _network_lock:
        return _build_network()


@lru_cache(maxsize=1)
def _build_network() -> _Network:
    # The whole city, not the k-hop training subgraph: the subgraph exists because the
    # STGNN needs a dense adjacency, a constraint routing does not share. Risk is still
    # only known on the camera-covered edges -- the rest fall back to `missing_risk`.
    graph = build_full_graph(SPATIAL_DIR)
    risk_edges_path = Path(settings.risk_edges_path)
    if not risk_edges_path.is_absolute():
        risk_edges_path = REPO_ROOT / risk_edges_path

    # Read once and split by window; each window's weighted graph is only built when a
    # departure first lands on it (_weighted_graphs), since aligning one window to the
    # whole-city edge list takes a fraction of a second and there are hundreds.
    risk_by_window: Dict[str, pd.DataFrame] = {}
    risk_low = risk_high = float("nan")
    if risk_edges_path.exists():
        frame = load_risk_edges(risk_edges_path)
        keys = window_key(frame)
        risk_by_window = {str(k): rows for k, rows in frame[RISK_KEY + ["risk"]].groupby(keys, sort=True)}
        # Terciles over the edges the decoder actually scored. Taken over every edge
        # they would mostly measure the imputed constant, and every route would come
        # back the same colour. Pooled over all windows rather than per window, so that
        # leaving at a heavy time of day reads as heavier than leaving at a light one --
        # per-window cutoffs would split every departure time into the same three shares.
        risk_low, risk_high = (float(x) for x in np.percentile(frame["risk"].to_numpy(), [33, 66]))
    else:
        logger.warning("%s not found: routing without congestion risk", risk_edges_path)

    free_flow_seconds = None
    if (SPATIAL_DIR / "metro_manila_travel_time.npz").exists():
        free_flow_seconds = load_free_flow_seconds(SPATIAL_DIR, graph)

    node_coords = _load_node_coords(SPATIAL_DIR, graph.node_ids)

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
        graph, risk_by_window, free_flow_seconds, node_coords, risk_low, risk_high,
        events, event_intersections, _load_edge_names(SPATIAL_DIR, graph),
    )


@lru_cache(maxsize=16)
def _weighted_graphs(window: Optional[str]) -> Tuple[WeightedGraph, WeightedGraph]:
    """(ANTRoute, baseline) weighted graphs for one recorded window. With no window (no
    risk_edges.csv) every edge carries zero risk, so both reduce to the plain road graph
    and the ETA to free-flow time."""
    net = _network()
    if window is None:
        wg = build_weighted_graph(net.graph, np.zeros(net.graph.edge_index.shape[1]), coverage=0.0, scored_edges=0)
        return wg, wg.with_lambda(0.0)
    rows = net.risk_by_window[window]
    # What to assume for an edge no camera informs. 0 would be actively wrong here: a
    # scored edge at the observed mean risk costs ~2.1x its length at lambda=2, so an
    # unscored edge assumed risk-free would look less than half the price, and the
    # router would systematically abandon the monitored corridor for roads it knows
    # nothing about. Assuming this window's observed mean instead makes "unknown" cost
    # the same as "typical", so coverage gaps neither attract nor repel a route.
    missing_risk = float(rows["risk"].mean()) if len(rows) else 0.5
    risk, scored = risk_vector(net.graph, rows, missing_risk)
    wg_antroute = build_weighted_graph(
        net.graph,
        risk,
        lam=DEFAULT_LAMBDA,
        coverage=scored / net.graph.edge_index.shape[1],
        scored_edges=scored,
        window=window,
    )
    # lambda = 0: distance-only, same ACO search otherwise.
    return wg_antroute, wg_antroute.with_lambda(0.0)


def window_for_departure(depart_at: datetime) -> Tuple[Optional[str], str]:
    """
    The recorded window standing in for a departure at `depart_at` (naive Manila time),
    and a line telling the user which recorded traffic that is. See
    match_recorded_window. No window (None) when risk_edges.csv isn't loaded.
    """
    windows = _network().risk_by_window
    if not windows:
        return None, NO_RISK_NOTE
    window, covered = match_recorded_window(windows.keys(), depart_at)
    return window, describe_window(window, covered, depart_at)


def describe_window(window: str, covered: bool, depart_at: datetime) -> str:
    """One line telling the user which recorded traffic their route was planned on."""
    recorded = datetime.fromisoformat(window)
    when = f"{recorded:%a %b} {recorded.day}, {_clock(recorded)}"
    if covered:
        return f"Traffic based on recorded data from {when}"
    return (
        f"No traffic data recorded around {_clock(depart_at)} - using the closest recorded "
        f"peak period ({when})"
    )


def _clock(when: datetime) -> str:
    return f"{when.hour % 12 or 12}:{when.minute:02d} {'AM' if when.hour < 12 else 'PM'}"


def _as_of(window: Optional[str]) -> Optional[datetime]:
    """
    The moment the system is routing "as of".

    There is no live feed, so this is the timestamp of the recorded window the trip's
    departure was matched to, which keeps the predicted-congestion and incident signals
    on the same clock. EVENT_AS_OF overrides it, which is what a demo uses to sit at a
    recorded moment that has incidents on the board.
    """
    override = (settings.event_as_of or "").strip()
    raw = override or window or ""
    if not raw:
        return None
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=ZoneInfo(LOCAL_TZ))


def live_events(net: _Network, window: Optional[str]) -> List[tuple]:
    """(event, strength) pairs in effect at the moment this window represents."""
    when = _as_of(window)
    if when is None or not net.events:
        return []
    return active_at(net.events, when)


def available() -> bool:
    """False (rather than raising) when the road network files under
    data/processed/spatial aren't present, so nothing can be routed at all."""
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
    if not net.risk_by_window:
        return "unknown"  # no risk_edges.csv: nothing to say, rather than a false "clear"
    if mean_risk <= net.risk_low:
        return "clear"
    if mean_risk <= net.risk_high:
        return "moderate"
    return "heavy"


def _path_points(net: _Network, node_ids: List[int]) -> List[dict]:
    return [{"lat": net.node_coords[n][0], "lng": net.node_coords[n][1]} for n in node_ids]


def _via_roads(net: _Network, wg: WeightedGraph, nodes: List[int], max_roads: int = 3) -> Optional[str]:
    """
    The roads a route mostly runs on: the `max_roads` named roads it covers the most
    distance on, in the order it reaches them -- e.g. "Shaw Boulevard, Epifanio de los
    Santos Avenue, Ortigas Avenue". None if road names aren't loaded or none are named.
    """
    if net.edge_names is None:
        return None
    indices = [wg.index_of(n) for n in nodes]
    distance: Dict[str, float] = {}
    for u, v in zip(indices, indices[1:]):
        e = wg.edge_id(u, v)
        name = net.edge_names[e]
        if name:
            distance[name] = distance.get(name, 0.0) + float(wg.distance[e])  # dict keeps first-seen order
    main = set(sorted(distance, key=distance.get, reverse=True)[:max_roads])
    return ", ".join(name for name in distance if name in main) or None


def _metrics_to_option(net: _Network, wg: WeightedGraph, label: str, m: PathMetrics) -> dict:
    eta_seconds = (
        path_eta_seconds(wg, net.free_flow_seconds, m.nodes) if net.free_flow_seconds is not None else None
    )
    if eta_seconds is not None and not net.risk_by_window:
        # No congestion risk loaded: every edge is at zero risk, which alone would be
        # an empty-road ETA. Use the city's typical congestion instead.
        eta_seconds *= TYPICAL_CONGESTION_MULTIPLIER
    duration_min = (
        max(1, round(eta_seconds / 60))
        if eta_seconds is not None
        else max(1, round(m.distance_m / 1000 / FALLBACK_SPEED_KMH * 60))
    )
    return {
        "label": label,
        "via": _via_roads(net, wg, m.nodes) or "local roads",
        "duration_min": duration_min,
        "distance_km": round(m.distance_m / 1000, 2),
        "congestion_level": _congestion_label(net, m.mean_risk),
        "event_note": None,
        "path": _path_points(net, m.nodes),
    }


def _event_note(
    wg_risk_only: WeightedGraph,
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
        without = ant_colony_shortest_path(wg_risk_only, stops[0], stops[1], config).best
    except (ValueError, KeyError):
        return None

    indices = [wg_risk_only.index_of(n) for n in without.nodes]
    edges = [wg_risk_only.edge_id(u, v) for u, v in zip(indices, indices[1:])]
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
    window: Optional[str] = None,
) -> List[dict]:
    """
    Real risk-aware (or distance-only baseline) routes for the given stops, in the
    same dict shape route_engine.plan_routes() returns, on the congestion recorded in
    `window` (from window_for_departure; None routes without congestion risk).

    Raises RouteOutsideNetworkError if any stop is too far from every road in the
    network, and NoRouteError if no drivable path joins the stops.

    Returns up to three routes, shortest first under the model's cost (see
    aco_routing.diverse_routes); fewer when the roads offer no separate alternative.
    """
    net = _network()
    wg_antroute, wg_baseline = _weighted_graphs(window)
    wg = wg_antroute if model == "antroute" else wg_baseline

    # Event-awareness is what ANTROUTE adds over the baseline: the baseline routes on
    # distance alone, with neither predicted congestion nor reported incidents, so it
    # stays the honest "no model at all" comparison.
    impact = None
    live: List[tuple] = []
    if model == "antroute":
        live = live_events(net, window)
        if live:
            impact = event_impact(net.graph, live, net.event_intersections)
            wg = apply_events(wg, impact, mu=settings.event_mu)

    stops = [nearest_node(net, origin_coords)] + [nearest_node(net, c) for c in destination_coords]
    if any(a == b for a, b in zip(stops, stops[1:])):
        raise NoRouteError("Two consecutive stops are at the same spot on the road network.")

    # Lighter than the library default (20 ants x 60 iterations) because the colony now
    # searches the whole city. With the goal-directed heuristic the ants converge on the
    # optimum almost immediately -- measured on a 165-hop cross-city route, 8x15 returns
    # the identical path as 20x60 in 0.9s instead of 8.7s -- so the extra tours buy
    # response time, not route quality. diverse_routes runs the colony up to 4 times
    # for alternatives, which is the other reason to keep each run light.
    config = AntColonyConfig(n_ants=8, n_iterations=15, seed=0)
    # The three shortest genuinely different routes under this model's cost -- for
    # ANTRoute the risk- and incident-weighted W, for the baseline plain distance.
    try:
        routes = diverse_routes(wg, stops, k=3, config=config)
    except ValueError as exc:
        logger.info("No route for stops %s: %s", stops, exc)
        raise NoRouteError("No drivable route was found between these places.") from exc

    options = [
        _metrics_to_option(net, wg, "Best route" if i == 0 else f"Alternative {i}", m)
        for i, m in enumerate(routes)
    ]
    note = _event_note(wg_antroute, impact, live, stops, config)
    if note:
        options[0]["event_note"] = note
    return options
