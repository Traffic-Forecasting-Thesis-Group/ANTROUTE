"""
Wires the real CNN+LSTM -> RADR STGNN -> MLP Decoder -> Dynamic Weight Engine -> ACO
pipeline into the FastAPI route-planning endpoint, alongside the thesis baseline,
Improved ACO (Cheng 2023).

No live data feed exists (CCTV/Twitter/weather are all historical), so a trip is
routed against the recorded window in risk_edges.csv that matches its departure time
-- same time of day, preferring the same weekday (departure_window.
match_recorded_window). Only the AM and PM peaks were recorded; a departure outside
them uses the closest peak window and says so in the response. Both systems use that
same window: ANTROUTE its predicted risk, the baseline the camera observations of it.

risk_edges.csv is not in git (~100MB). Without it ANTROUTE's routes are still real --
the same road graph, ACO search and free-flow ETAs -- just with no congestion risk, and
the response says so rather than passing that off as traffic-aware.

Routing covers the whole Metro Manila road network (build_full_graph), not just the
k-hop subgraph the STGNN trains on -- that subgraph exists only because the model
needs a dense adjacency, a constraint routing does not share. Congestion risk is only
predicted on the subgraph; every other edge is routed and timed on road distance and
speed limit alone (risk 0, as the thesis scope states), the same rule
scripts/evaluate_routing.py uses. An origin or destination further than MAX_SNAP_KM
from every road in the network raises RouteOutsideNetworkError rather than silently
inventing a route.
"""

from __future__ import annotations

import csv
import json
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
from src.data.alignment import WINDOW_STEPS
from src.data.graph_data import GraphData, build_full_graph
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path, diverse_routes
from src.routing.baseline_iaco import IacoConfig, IacoGraph
from src.routing.baseline_router import (
    IACO,
    SHORTEST_DISTANCE,
    BaselineInputs,
    build_inputs,
    camera_flow,
    camera_observations,
    load_vehicle_counts,
    plan_baseline,
)
from src.routing.departure_window import match_recorded_window
from src.routing.dynamic_weight import (
    DEFAULT_LAMBDA,
    DEFAULT_MISSING_RISK,
    RISK_KEY,
    NearestScored,
    PathMetrics,
    WeightedGraph,
    build_weighted_graph,
    MAX_FILL_M,
    fill_unscored_risk,
    load_risk_edges,
    risk_vector,
    scored_mask,
    window_key,
)
from src.routing.eta_engine import load_free_flow_seconds, path_eta_seconds
from src.routing.event_layer import (
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

NO_RISK_NOTE = "Congestion data is not loaded on the server - ETA uses road speed limits only"

# The baseline colony: Cheng's parameters (alpha 1, beta 5, rho 0.3, q0 0.7 for a large
# network) with dead-end recovery, at the same 8 ants x 15 iterations the app gives
# ANTROUTE (see plan_real_routes), searched in a corridor around each leg. On the whole
# city at baseline_iaco's default 20 x 60, ants wander thousands of steps on a ~100-step
# trip and one request ran over an hour. scripts/evaluate_routing.py keeps those defaults.
BASELINE_CONFIG = IacoConfig(n_ants=8, n_iterations=15, seed=0)
BASELINE_CORRIDOR_MARGIN = 0.2  # ellipse 20% longer than the straight line, plus 500 m


class RouteOutsideNetworkError(Exception):
    """Origin or destination is too far from any road in the network."""


class NoRouteError(Exception):
    """The stops are on the network but no drivable path joins them."""


@dataclass
class _Network:
    graph: GraphData
    # window_start -> that window's decoder rows (and camera observations, weak_target);
    # empty without risk_edges.csv
    risk_by_window: Dict[str, pd.DataFrame]
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
        # Snapping a searched place scans every node, and the routable graph is the whole
        # city (59,521 nodes), so keep the coordinates as arrays and let numpy do it
        # rather than looping in Python on every request.
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
    # Building takes ~40s, and main.py starts it in a background thread at startup. The
    # lock makes a request that arrives meanwhile wait for that build instead of starting
    # a second one.
    with _network_lock:
        return _build_network()


@lru_cache(maxsize=1)
def _build_network() -> _Network:
    # The whole city, not the k-hop training subgraph: the subgraph exists because the
    # STGNN needs a dense adjacency, a constraint routing does not share.
    graph = build_full_graph(SPATIAL_DIR)
    edges_path = risk_edges_path()

    # Read once and split by window; each window's weighted graph is only built when a
    # departure first lands on it (_weighted_graph), since aligning one window to the
    # whole-city edge list takes a fraction of a second and there are hundreds. The
    # camera observations (weak_target) ride along for the baseline.
    risk_by_window: Dict[str, pd.DataFrame] = {}
    risk_low = risk_high = float("nan")
    if edges_path.exists():
        frame = load_risk_edges(edges_path)
        keys = window_key(frame)
        columns = RISK_KEY + ["risk"] + (["weak_target"] if "weak_target" in frame.columns else [])
        risk_by_window = {str(k): rows for k, rows in frame[columns].groupby(keys, sort=True)}
        # Terciles over the edges the decoder actually scored. Taken over every edge
        # they would mostly measure the imputed constant, and every route would come
        # back the same colour. Pooled over all windows rather than per window, so that
        # leaving at a heavy time of day reads as heavier than leaving at a light one --
        # per-window cutoffs would split every departure time into the same three shares.
        risk_low, risk_high = (float(x) for x in np.percentile(frame["risk"].to_numpy(), [33, 66]))
    else:
        logger.warning("%s not found: routing without congestion risk", edges_path)

    free_flow_seconds = None
    if (SPATIAL_DIR / "metro_manila_travel_time.npz").exists():
        free_flow_seconds = load_free_flow_seconds(SPATIAL_DIR, graph)

    node_coords = _load_node_coords(SPATIAL_DIR, graph.node_ids)

    # Rule-based incident layer (src/routing/event_layer.py). Not part of the thesis
    # method -- there, event text reaches routing only through the learned DistilBERT ->
    # CNN+LSTM -> RADR STGNN path, i.e. the risk scores above -- so it is off unless
    # EVENT_MU is set above 0, and the thesis evaluation never uses it.
    events: List[TrafficEvent] = []
    raw_twitter = REPO_ROOT / "data/raw/twitter"
    landmarks = load_landmarks(REPO_ROOT / "configs/event_landmarks.csv")
    if settings.event_mu > 0 and raw_twitter.exists() and landmarks:
        events = parse_alerts(raw_twitter, landmarks)

    # Where an incident label places on the graph. Starts from the camera intersections
    # (already graph indices) and adds the non-camera EDSA landmarks the MMDA feed also
    # names often -- Guadalupe, Magallanes, Balintawak, etc. -- geocoded once and snapped
    # to the nearest real node (configs/event_intersections.csv records the source
    # coordinates and the snap, so a bad match is auditable).
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


def risk_edges_path() -> Path:
    """Where this server reads risk_edges.csv from (app setting RISK_EDGES_PATH)."""
    path = Path(settings.risk_edges_path)
    return path if path.is_absolute() else REPO_ROOT / path


def model_provenance() -> dict:
    """
    Which scored artefact -- and therefore which trained checkpoint -- this running server is
    actually serving routes from (GET /health/model).

    risk_edges.csv is too large to commit, so it is copied in by hand from wherever the
    scoring run wrote it; nothing in the file name says which checkpoint produced it, and a
    stale copy produces perfectly plausible routes. During a demo "which model is this?" has
    to be answerable from the system itself.

    The checkpoint name comes from risk_summary.json, which predict_congestion_risk.py writes
    beside risk_edges.csv; so do `context` (True for a multimodal checkpoint, whose risk moves
    on roads without a camera) and `context_drop`. If it was not copied across, that is
    reported as unknown rather than guessed at. Raises FileNotFoundError when there is no
    risk_edges.csv at all.
    """
    path = risk_edges_path()
    if not path.exists():
        raise FileNotFoundError(str(path))
    net = _network()
    stat = path.stat()
    window, _ = window_for_departure(datetime.now(ZoneInfo(LOCAL_TZ)).replace(tzinfo=None, microsecond=0))
    wg = _weighted_graph(window)
    out = {
        "risk_edges": {
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, ZoneInfo(LOCAL_TZ)).isoformat(),
            "windows": len(net.risk_by_window),
        },
        "serving_window": window,
        "graph": {
            "nodes": int(net.graph.n_nodes),
            "edges": int(wg.n_edges),
            "edges_with_a_predicted_risk": int(wg.scored_edges),
            "coverage": round(float(wg.coverage), 4),
        },
        "events": {"parsed": len(net.events), "placed_intersections": len(net.event_intersections)},
        "checkpoint": "unknown -- risk_summary.json was not copied beside risk_edges.csv",
        "calibration": "none -- no risk_calibration.json beside risk_edges.csv",
    }

    summary_path = path.with_name("risk_summary.json")
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            out["checkpoint"] = f"risk_summary.json present but unreadable: {exc}"
        else:
            out["checkpoint"] = summary.get("checkpoint", out["checkpoint"])
            out["scoring_run"] = {
                k: summary[k]
                for k in ("windows", "sessions", "edges", "camera_edges", "context", "context_drop",
                          "flood_hazard", "use_text", "splits")
                if k in summary
            }

    calibration_path = path.with_name("risk_calibration.json")
    if calibration_path.exists():
        try:
            out["calibration"] = json.loads(calibration_path.read_text(encoding="utf-8")).get("calibration")
        except (OSError, ValueError) as exc:
            out["calibration"] = f"risk_calibration.json present but unreadable: {exc}"
    return out


@lru_cache(maxsize=16)
def _weighted_graph(window: Optional[str]) -> WeightedGraph:
    """
    ANTROUTE's weighted graph (lambda = DEFAULT_LAMBDA) for one recorded window.

    risk_edges.csv scores only the camera subgraph, ~2% of the city's edges. The rest are
    filled by fill_unscored_risk -- the nearest scored road's risk within MAX_FILL_M, else the
    window's median -- rather than DEFAULT_MISSING_RISK (0): with 0, a road without a
    prediction looked empty, and ANTROUTE preferred side streets with no data over EDSA. The
    WeightedGraph's coverage / scored_edges still count only the decoder's own predictions.
    With no window (no risk_edges.csv) every edge carries 0.
    """
    net = _network()
    n_edges = net.graph.edge_index.shape[1]
    if window is None:
        return build_weighted_graph(net.graph, np.zeros(n_edges), coverage=0.0, scored_edges=0)
    frame = net.risk_by_window[window]
    risk, scored = risk_vector(net.graph, frame, DEFAULT_MISSING_RISK)
    mask = scored_mask(net.graph, frame)
    risk, _ = fill_unscored_risk(net.graph, risk, mask, _nearest_scored(mask.tobytes()))
    return build_weighted_graph(
        net.graph, risk, lam=DEFAULT_LAMBDA, coverage=scored / n_edges, scored_edges=scored, window=window
    )


@lru_cache(maxsize=2)
def _nearest_scored(mask_bytes: bytes) -> NearestScored:
    """NearestScored for one set of scored edges -- the same subgraph in every window, so one
    road-distance pass serves them all."""
    mask = np.frombuffer(mask_bytes, dtype=bool)
    return NearestScored.build(_network().graph, mask)


@lru_cache(maxsize=16)
def _baseline_inputs(window: Optional[str]) -> BaselineInputs:
    """
    What the baseline plans with at one recorded window: Cheng's cost on the same city
    graph ANTROUTE routes on, from what the cameras observed then (see
    src/routing/baseline_router.py). No window means no observations, so free-flow travel
    time everywhere -- what the cameras would report on an empty road.
    """
    net = _network()
    wg = _weighted_graph(window)
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
    rows = net.risk_by_window.get(window) if window else None
    observed = {}
    if rows is not None and "weak_target" in rows.columns:
        observed = camera_observations(
            rows[rows["weak_target"].notna()],
            np.asarray(net.graph.node_ids),
            list(net.graph.camera_nodes.values()),
        )
    # Traffic flow n_ij: mean YOLO vehicle count per camera over the window, from the
    # auto_labels.csv files in VEHICLE_COUNTS_DIR -- the same rule scripts/evaluate_routing.py
    # applies. Without those files the flow term is zero (BaselineInputs.flow_observed=False).
    flow = None
    counts_dir = Path(settings.vehicle_counts_dir)
    if not counts_dir.is_absolute():
        counts_dir = REPO_ROOT / counts_dir
    paths = sorted(counts_dir.glob("*.csv")) if counts_dir.exists() else []
    if paths and window:
        start = pd.Timestamp(window)
        counts = load_vehicle_counts(paths, net.graph.camera_nodes, REPO_ROOT / "configs/camera_nodes.csv")
        flow = camera_flow(counts, start, start + pd.Timedelta(minutes=WINDOW_STEPS))
    return build_inputs(g, free_flow, observed, camera_flow=flow)


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


# A route's congestion is reported only when most of its length has a risk that came from
# the model -- predicted on the road itself, or borrowed from a predicted road within
# MAX_FILL_M. Below this share the risk is mostly the window's median stand-in, and calling
# that "moderate" would present a default as an assessment.
MIN_RISK_COVERAGE = 0.5


def _congestion_label(net: _Network, mean_risk: float, coverage: float = 1.0) -> str:
    if not net.risk_by_window:
        return "unknown"  # no risk_edges.csv: nothing to say, rather than a false "clear"
    if coverage < MIN_RISK_COVERAGE:
        return "unknown"  # no prediction near this route: the risk is the window median
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


def _duration_min(net: _Network, wg: WeightedGraph, m: PathMetrics) -> int:
    """
    Travel time of a route by the app's one ETA engine, whichever system chose the route.

    Both systems' cards are timed with this, on ANTROUTE's risk graph for the same window,
    so two systems that pick the same road always show the same time, and a difference on
    the map is a difference in the route. Each system's own ETA prediction is a separate
    question, scored offline against Apple Maps by scripts/evaluate_routing.py.
    """
    if net.free_flow_seconds is None:
        return max(1, round(m.distance_m / 1000 / FALLBACK_SPEED_KMH * 60))
    return max(1, round(path_eta_seconds(wg, net.free_flow_seconds, m.nodes) / 60))


def risk_coverage(wg: WeightedGraph, nodes: List[int]) -> float:
    """
    Share of a route's length whose risk came from the model for wg's window: predicted on
    the edge, or borrowed by fill_unscored_risk from a predicted road within MAX_FILL_M. The
    rest carries the window's median as a stand-in. 0 without a window.
    """
    if wg.window is None:
        return 0.0
    informed = _informed_edges(wg.window)
    indices = [wg.index_of(n) for n in nodes]
    edges = [wg.edge_id(u, v) for u, v in zip(indices, indices[1:])]
    length = wg.distance[edges]
    return float(length[informed[edges]].sum() / length.sum()) if length.sum() > 0 else 0.0


@lru_cache(maxsize=16)
def _informed_edges(window: str) -> np.ndarray:
    """[E] True where this window's risk came from the model rather than the median stand-in."""
    net = _network()
    mask = scored_mask(net.graph, net.risk_by_window[window])
    nearest = _nearest_scored(mask.tobytes())
    src, dst = net.graph.edge_index[0].numpy(), net.graph.edge_index[1].numpy()
    near = np.minimum(nearest.distance_m[src], nearest.distance_m[dst]) <= MAX_FILL_M
    return mask | near


def _metrics_to_option(net: _Network, wg: WeightedGraph, label: str, m: PathMetrics) -> dict:
    coverage = risk_coverage(wg, m.nodes)
    return {
        "label": label,
        "via": _via_roads(net, wg, m.nodes) or "local roads",
        "duration_min": _duration_min(net, wg, m),
        "distance_km": round(m.distance_m / 1000, 2),
        "congestion_level": _congestion_label(net, m.mean_risk, coverage),
        "mean_risk": round(m.mean_risk, 4),
        "risk_coverage": round(coverage, 3),
        "algorithm": "antroute",
        "fallback_reason": None,
        "event_note": None,
        "path": _path_points(net, m.nodes),
        # Graph node ids along the route, for scripts/citywide_trips.py; not part of the API
        # response (RouteOption has no such field, so the endpoint drops it).
        "nodes": [int(n) for n in m.nodes],
    }


def _baseline_option(net: _Network, window: Optional[str], stops: List[int]) -> dict:
    """
    The baseline's single route (IACO returns one optimal path), in the same dict shape as
    ANTROUTE's options.

    Distance, risk and duration are all measured the same way as ANTROUTE's (on its graph
    for the same window, by _duration_min), so the two cards compare routes, not measuring
    methods. Raises baseline_router.BaselineNoRouteError when IACO finds no route and the
    fallback is off.
    """
    route = plan_baseline(
        _baseline_inputs(window),
        stops,
        BASELINE_CONFIG,
        fallback=settings.baseline_fallback,
        corridor_margin=BASELINE_CORRIDOR_MARGIN,
    )
    wg = _weighted_graph(window)
    m = wg.evaluate_path(route.nodes)
    if route.algorithm == IACO:
        method = "Improved ACO (Cheng 2023)"
    elif route.algorithm == SHORTEST_DISTANCE:
        method = "Shortest distance (IACO found no route)"
    else:
        method = f"IACO for {route.legs_by_iaco} of {route.legs} legs, shortest distance for the rest"
    roads = _via_roads(net, wg, m.nodes)
    coverage = risk_coverage(wg, m.nodes)
    # The app already names the method under every baseline card ("Baseline route · Improved
    # ACO (Cheng 2023)"), so `via` adds it only when it is NOT plain IACO -- a fallback is the
    # one thing the card would otherwise hide.
    if roads and route.algorithm == IACO:
        via = roads
    else:
        via = f"{roads} ({method})" if roads else method
    return {
        "label": "Baseline route",
        "via": via,
        "duration_min": _duration_min(net, wg, m),
        "distance_km": round(m.distance_m / 1000, 2),
        "congestion_level": _congestion_label(net, m.mean_risk, coverage),
        "mean_risk": round(m.mean_risk, 4),
        "risk_coverage": round(coverage, 3),
        "algorithm": route.algorithm,
        "fallback_reason": route.fallback_reason,
        "event_note": None,
        "path": _path_points(net, route.nodes),
        "nodes": [int(n) for n in route.nodes],  # see _metrics_to_option
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


def snap_stops(points: List[Tuple[float, float]]) -> List[int]:
    """The graph node each stop (lat, lng) routes from, origin first -- as plan_real_routes snaps them."""
    net = _network()
    return [nearest_node(net, p) for p in points]


def plan_real_routes(
    origin_coords: Tuple[float, float],
    destination_coords: List[Tuple[float, float]],
    model: str = "antroute",
    window: Optional[str] = None,
) -> List[dict]:
    """
    Routes for the given stops, in the same dict shape route_engine.plan_routes() returns,
    on the traffic recorded in `window` (from window_for_departure; None routes without
    congestion risk).

    model="antroute" is the risk-aware ACO: up to three routes, best first under its cost
    (see aco_routing.diverse_routes), fewer when the roads offer no separate alternative.
    model="baseline" is the thesis baseline, Improved ACO (Cheng 2023), with its one
    optimal path. See _baseline_option() and src/routing/baseline_router.py.

    Raises RouteOutsideNetworkError if any stop is too far from every road in the
    network, NoRouteError if no drivable path joins the stops, and
    baseline_router.BaselineNoRouteError when the baseline finds none.
    """
    net = _network()
    stops = snap_stops([origin_coords, *destination_coords])
    if any(a == b for a, b in zip(stops, stops[1:])):
        raise NoRouteError("Two consecutive stops are at the same spot on the road network.")

    # The baseline plans on observed traffic with Cheng's own cost and colony. It sees neither
    # ANTROUTE's predicted risk nor the reported incidents: those are what ANTROUTE adds.
    if model == "baseline":
        return [_baseline_option(net, window, stops)]

    wg_risk_only = _weighted_graph(window)
    wg = wg_risk_only
    impact = None
    live = live_events(net, window) if settings.event_mu > 0 else []
    if live:
        impact = event_impact(net.graph, live, net.event_intersections)
        wg = apply_events(wg, impact, mu=settings.event_mu)

    # Lighter than the library default (20 ants x 60 iterations) because the colony
    # searches the whole city. With the goal-directed heuristic the ants converge on the
    # optimum almost immediately -- measured on a 165-hop cross-city route, 8x15 returns
    # the identical path as 20x60 in 0.9s instead of 8.7s -- so the extra tours buy
    # response time, not route quality. diverse_routes runs the colony up to 4 times
    # for alternatives, which is the other reason to keep each run light.
    config = AntColonyConfig(n_ants=8, n_iterations=15, seed=0)
    try:
        routes = diverse_routes(wg, stops, k=3, config=config)
    except ValueError as exc:
        logger.info("No route for stops %s: %s", stops, exc)
        raise NoRouteError("No drivable route was found between these places.") from exc

    options = [
        _metrics_to_option(net, wg_risk_only, "Best route" if i == 0 else f"Alternative {i}", m)
        for i, m in enumerate(routes)
    ]
    note = _event_note(wg_risk_only, impact, live, stops, config)
    if note:
        options[0]["event_note"] = note
    return options
