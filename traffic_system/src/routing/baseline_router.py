"""
The thesis baseline adapted to ANTROUTE's codebase: Improved ACO (Cheng 2023) planning on the
same road graph, at the same moment, as ANTROUTE.

What stays exactly as the paper wrote it lives in src/routing/baseline_iaco.py and is not
touched here: the AHP-weighted cost (eq. 4-6), direction guidance (eq. 8), the transfer rule
(eq. 10-11), local and global pheromone updates (eq. 12-14) and the tabu list. In particular
there is no backtracking and no dead-end pruning. An ant left with no unvisited road dies, as in
the paper, because that limitation is what ANTROUTE's search exists to overcome.

What is adapted, and only here, is the input data and the plumbing:

  - Travel time t_ij: free-flow time scaled by the congestion the cameras observed in the
    routing window (eta_engine.congested_eta), spread to every edge from the nearest camera.
    This is the same construction scripts/evaluate_baseline_iaco.py uses. It is observed, not
    predicted: the paper plans on measured traffic, so the baseline must not borrow ANTROUTE's
    risk model.
  - Traffic flow n_ij: YOLO vehicle counts per camera when they are supplied. Without them the
    flow term is zero, so s reduces to its length and time terms (w3 = 0.105 goes unused).
    BaselineInputs.flow_observed records which case applies.
  - Multi-stop trips are routed leg by leg, because the paper defines a single O-D search.

When no ant reaches the destination, the leg falls back to the spatial shortest-distance path,
which is the comparator the paper itself reports (Table 9). On the Metro Manila network that is
nearly every trip: in a 24-trip test all 28,800 ants died, after a median of 7 steps, at the
city's dead-end streets. The fallback keeps route optimality computable for every trip, while
BaselineRoute.algorithm keeps IACO's own completion rate visible as a separate result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.routing.baseline_iaco import (
    IacoConfig,
    IacoGraph,
    comprehensive_cost,
    fill_from_nearest_camera,
    iaco_search,
    spatial_shortest_path,
)
from src.routing.eta_engine import DEFAULT_GAMMA, congested_eta

IACO = "iaco"
SHORTEST_DISTANCE = "shortest_distance"
MIXED = "mixed"


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
        "ants reached the destination, because each was left with no unvisited road to take."
    )
    if fell_back:
        reason += " Showing the spatial shortest-distance route, the comparator the paper itself reports."
    return reason


def plan_baseline(
    inputs: BaselineInputs,
    stops: Sequence[int],
    config: IacoConfig = IacoConfig(),
    fallback: bool = True,
) -> BaselineRoute:
    """
    Route through `stops` (road-network node ids) leg by leg: IACO first, shortest distance
    when IACO fails and `fallback` is on.

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
        o, d = g.index_of(origin), g.index_of(destination)
        tau = np.full(g.n_edges, config.initial_pheromone, dtype=np.float64)
        found = iaco_search(g, o, d, inputs.s, tau, config, np.random.default_rng(config.seed))
        if found is not None:
            by_iaco += 1
            leg_nodes = [int(g.node_ids[i]) for i in found.path]
            leg_edges = list(found.edges)
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
