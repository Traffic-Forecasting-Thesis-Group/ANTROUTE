"""
The thesis baseline adapted to ANTROUTE's codebase: Improved ACO (Cheng 2023) planning on the
same road graph, at the same moment, as ANTROUTE.

What stays exactly as the paper wrote it lives in src/routing/baseline_iaco.py and is not
touched here: the AHP-weighted cost (eq. 4-6), direction guidance (eq. 8), the transfer rule
(eq. 10-11), local and global pheromone updates (eq. 12-14) and the tabu list. The one
adaptation to the algorithm, dead-end recovery, is documented there (IacoConfig.dead_end_recovery):
without it every ant died at Metro Manila's dead-end streets and the baseline degenerated into
shortest distance.

What is adapted here is the input data and the plumbing:

  - Travel time t_ij: free-flow time scaled by the congestion the cameras observed in the
    routing window (eta_engine.congested_eta), spread to every edge from the nearest camera.
    This is the same construction scripts/evaluate_baseline_iaco.py uses. It is observed, not
    predicted: the paper plans on measured traffic, so the baseline must not borrow ANTROUTE's
    risk model.
  - Traffic flow n_ij: the mean YOLO vehicle count per camera over the routing window
    (load_vehicle_counts / camera_flow below, read from the auto_labels.csv files the labeling
    step writes), spread to every edge from the nearest camera. Only when no counts are
    supplied is the flow term zero; BaselineInputs.flow_observed records which case applies.
  - Multi-stop trips are routed leg by leg, because the paper defines a single O-D search.
  - Optionally (corridor_margin) each leg is searched on the roads in an ellipse around it
    rather than the whole city (baseline_iaco.corridor_subgraph), which the live app needs
    to answer in seconds instead of hours.

When no ant reaches the destination the leg has no baseline route. The spatial shortest-distance
fallback (the paper's Table 9 comparator) is still available with fallback=True, but it is off
by default for the thesis comparison, because a shortest-distance route is not Improved ACO.
BaselineRoute.algorithm keeps IACO's own completion rate visible either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.routing.baseline_iaco import (
    IacoConfig,
    IacoGraph,
    comprehensive_cost,
    corridor_subgraph,
    fill_from_nearest_camera,
    iaco_search,
    spatial_shortest_path,
)
from src.data.graph_data import resolve_camera_map
from src.routing.eta_engine import DEFAULT_GAMMA, congested_eta

LOCAL_TZ = "Asia/Manila"

IACO = "iaco"
SHORTEST_DISTANCE = "shortest_distance"
MIXED = "mixed"

# Past this, a corridor is most of the city; search the whole graph instead.
MAX_CORRIDOR_MARGIN = 3.2


class BaselineNoRouteError(ValueError):
    """IACO found no route and the shortest-distance fallback is switched off."""


@dataclass(frozen=True)
class BaselineInputs:
    """Everything IACO plans with for one moment in time."""

    graph: IacoGraph
    s: np.ndarray               # [E] comprehensive cost, eq. 6
    travel_time: np.ndarray     # [E] seconds; IACO's own estimate of the trip time
    cameras_observed: int       # cameras with an observed congestion level this window
    flow_observed: bool         # False: no vehicle counts, the flow term of s is zero


@dataclass(frozen=True)
class BaselineRoute:
    nodes: List[int]                # road-network node ids, origin first
    edges: List[int]                # graph edge indices
    algorithm: str                  # IACO, SHORTEST_DISTANCE, or MIXED across legs
    legs_by_iaco: int
    legs: int
    travel_time_s: float            # sum of the t_ij IACO planned with
    fallback_reason: Optional[str]  # set whenever any leg is not IACO's


def camera_observations(
    rows: pd.DataFrame,
    node_ids: np.ndarray,
    camera_indices: Sequence[int],
    column: str = "weak_target",
) -> Dict[int, float]:
    """
    Mean observed value per camera for one window, keyed by graph index.

    An edge belongs to a camera when exactly one of its endpoints is a camera node, the same
    ownership rule scripts/evaluate_baseline_iaco.py applies. Rows with no observation are
    ignored, and so are edges that touch no camera or touch two.
    """
    cameras = {int(c) for c in camera_indices}
    position = {int(n): i for i, n in enumerate(node_ids)}
    labelled = rows[rows[column].notna()]
    totals: Dict[int, List[float]] = {}
    for u, v, value in zip(labelled["source_node_id"], labelled["target_node_id"], labelled[column]):
        iu, iv = position.get(int(u)), position.get(int(v))
        if iu is None or iv is None or (iu in cameras) == (iv in cameras):
            continue
        totals.setdefault(iu if iu in cameras else iv, []).append(float(value))
    return {camera: float(np.mean(values)) for camera, values in totals.items()}


def _local_naive(values: pd.Series) -> pd.Series:
    """Timestamps as naive Manila local time."""
    ts = pd.to_datetime(values, format="ISO8601")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return ts


def load_vehicle_counts(
    paths: Sequence[Path], camera_nodes: Mapping[str, int], camera_csv: Optional[Path]
) -> pd.DataFrame:
    """
    YOLO vehicle counts per frame (auto_labels.csv: camera_id, timestamp, n_vehicles), placed
    at the graph index of each camera's intersection. Columns: camera, time, n_vehicles.
    Frames from cameras that map to no intersection are dropped.
    """
    frames = [pd.read_csv(p, usecols=["camera_id", "timestamp", "n_vehicles"]) for p in paths]
    if not frames:
        return pd.DataFrame(columns=["camera", "time", "n_vehicles"])
    counts = pd.concat(frames, ignore_index=True)
    mapping, unmapped = resolve_camera_map(sorted(counts["camera_id"].unique()), list(camera_nodes), camera_csv)
    if unmapped:
        print(f"WARNING: {len(unmapped)} camera id(s) not mapped to an intersection, ignored: {unmapped[:5]}")
    counts["camera"] = counts["camera_id"].map(lambda c: camera_nodes.get(mapping.get(c), -1))
    counts = counts[counts["camera"] >= 0].copy()
    counts["time"] = _local_naive(counts["timestamp"])
    return counts[["camera", "time", "n_vehicles"]].sort_values("time").reset_index(drop=True)


def camera_flow(counts: pd.DataFrame, start, end) -> Dict[int, float]:
    """Mean vehicle count per camera (graph index) over [start, end)."""
    start = _local_naive(pd.Series([start])).iloc[0]
    end = _local_naive(pd.Series([end])).iloc[0]
    inside = counts[(counts["time"] >= start) & (counts["time"] < end)]
    return {int(c): float(v) for c, v in inside.groupby("camera")["n_vehicles"].mean().items()}


def build_inputs(
    graph: IacoGraph,
    free_flow_seconds: np.ndarray,
    camera_congestion: Mapping[int, float],
    camera_flow: Optional[Mapping[int, float]] = None,
    gamma: float = DEFAULT_GAMMA,
) -> BaselineInputs:
    """
    The paper's t_ij, n_ij and s_ij for every edge, from what the cameras observed.

    No congestion observation at all means free-flow travel time everywhere, which is what
    the cameras would report on an empty road. It is not an error.
    """
    free_flow_seconds = np.asarray(free_flow_seconds, dtype=np.float64)
    if free_flow_seconds.shape != (graph.n_edges,):
        raise ValueError(f"need one free-flow time per edge ({graph.n_edges}), received {free_flow_seconds.shape}")

    if camera_congestion:
        congestion = fill_from_nearest_camera(graph.src, graph.dst, graph.n_nodes, camera_congestion)
    else:
        congestion = np.zeros(graph.n_edges)
    travel_time = congested_eta(free_flow_seconds, congestion, gamma)

    flow_observed = bool(camera_flow)
    if flow_observed:
        flow = fill_from_nearest_camera(graph.src, graph.dst, graph.n_nodes, camera_flow)
    else:
        flow = np.zeros(graph.n_edges)

    return BaselineInputs(
        graph=graph,
        s=comprehensive_cost(graph.distance, travel_time, flow),
        travel_time=travel_time,
        cameras_observed=len(camera_congestion),
        flow_observed=flow_observed,
    )


def _no_route_reason(config: IacoConfig, failed_legs: int, legs: int, fell_back: bool) -> str:
    where = "" if legs == 1 else f" for {failed_legs} of {legs} legs"
    reason = (
        f"IACO (Cheng 2023) found no route{where}: none of its {config.n_ants * config.n_iterations} "
        "ants reached the destination."
    )
    if fell_back:
        reason += " Showing the spatial shortest-distance route, the comparator the paper itself reports."
    return reason


def _search_area(
    g: IacoGraph, origin: int, destination: int, margin: Optional[float]
) -> Tuple[IacoGraph, Optional[np.ndarray]]:
    """
    The graph a leg is searched on and, for a corridor, each of its edges' index in g
    (None when it is g itself). Widens the corridor until it joins the stops; when even
    the widest one does not, the whole graph.
    """
    if margin is None:
        return g, None
    o, d = g.index_of(origin), g.index_of(destination)
    while margin <= MAX_CORRIDOR_MARGIN:
        sub, edge_index = corridor_subgraph(g, o, d, margin)
        if sub.can_reach(sub.index_of(destination))[sub.index_of(origin)]:
            return sub, edge_index
        margin *= 2
    return g, None


def plan_baseline(
    inputs: BaselineInputs,
    stops: Sequence[int],
    config: IacoConfig = IacoConfig(),
    fallback: bool = False,
    corridor_margin: Optional[float] = None,
) -> BaselineRoute:
    """
    Route through `stops` (road-network node ids) leg by leg with IACO. When IACO finds no
    route, the leg raises BaselineNoRouteError, or falls back to shortest distance when
    `fallback` is on (not the thesis comparison; labelled as such in the result).

    With `corridor_margin`, each leg searches only the roads in an ellipse around it (see
    baseline_iaco.corridor_subgraph), doubling the margin until the ellipse connects the
    leg's stops. None searches the whole graph.

    Each leg starts from fresh pheromone and its own seeded generator, so a leg's result does
    not depend on how many legs came before it. Consecutive duplicate stops are skipped.
    """
    if len(stops) < 2:
        raise ValueError("a baseline route needs at least an origin and a destination")
    config.validate()
    g = inputs.graph

    legs = [(int(a), int(b)) for a, b in zip(stops, stops[1:]) if int(a) != int(b)]
    if not legs:
        raise ValueError("every stop is the same node; there is no trip to route")

    nodes: List[int] = [legs[0][0]]
    edges: List[int] = []
    by_iaco = 0
    for origin, destination in legs:
        search_g, edge_index = _search_area(g, origin, destination, corridor_margin)
        o, d = search_g.index_of(origin), search_g.index_of(destination)
        s = inputs.s if edge_index is None else inputs.s[edge_index]
        tau = np.full(search_g.n_edges, config.initial_pheromone, dtype=np.float64)
        found = iaco_search(search_g, o, d, s, tau, config, np.random.default_rng(config.seed))
        if found is not None:
            by_iaco += 1
            leg_nodes = [int(search_g.node_ids[i]) for i in found.path]
            leg_edges = list(found.edges) if edge_index is None else [int(edge_index[e]) for e in found.edges]
        elif fallback:
            leg_nodes = spatial_shortest_path(g, origin, destination)  # ValueError if unreachable
            leg_edges = g.edges_of([g.index_of(n) for n in leg_nodes])
        else:
            raise BaselineNoRouteError(_no_route_reason(config, 1, 1, fell_back=False))
        nodes.extend(leg_nodes[1:])
        edges.extend(leg_edges)

    if by_iaco == len(legs):
        algorithm, reason = IACO, None
    else:
        algorithm = SHORTEST_DISTANCE if by_iaco == 0 else MIXED
        reason = _no_route_reason(config, len(legs) - by_iaco, len(legs), fell_back=True)

    return BaselineRoute(
        nodes=nodes,
        edges=edges,
        algorithm=algorithm,
        legs_by_iaco=by_iaco,
        legs=len(legs),
        travel_time_s=float(inputs.travel_time[edges].sum()),
        fallback_reason=reason,
    )
