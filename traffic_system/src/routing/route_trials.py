"""
Turn ACO routes and their Apple Maps travel times into the trial records src.routing.metrics scores.

Scenarios follow the thesis (Section 3.8, Appendices 1-3):

    recommended        the best route ACO returns for an origin/destination
    alternative        the next-best route from the same ACO run
    multi_destination  a sequential multi-stop trip, routed leg by leg

Route Optimality (Equation 1) uses travel time as the cost, with Apple Maps as the oracle
on both sides so it measures route quality rather than how optimistic a system's ETA is:

    C_optimal    Apple Maps time of Apple's own route (summed over legs)
    C_predicted  Apple Maps time of the route the system chose (entered via waypoints)

The ETA metrics (Equations 2-6) compare the system's predicted time for its route with
Apple's time for that same route, per leg, as Appendix 2's note specifies.

Two systems are scored the same way:

    antroute   ACO on the risk-weighted graph (lambda > 0), risk-aware ETA (gamma > 0)
    baseline   the comparison ACO without event-derived risk (see evaluate_routing.py)
"""

from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
from src.routing.dynamic_weight import WeightedGraph
from src.routing.eta_engine import path_eta_seconds

ANTROUTE = "antroute"
BASELINE = "baseline"
SYSTEMS = (ANTROUTE, BASELINE)

RECOMMENDED = "recommended"
ALTERNATIVE = "alternative"
MULTI_DESTINATION = "multi_destination"
SCENARIOS = (RECOMMENDED, ALTERNATIVE, MULTI_DESTINATION)


def pick_waypoints(wg: WeightedGraph, nodes: Sequence[int], n: int) -> List[int]:
    """
    Up to `n` intermediate nodes spaced evenly by distance along the route, to enter as
    Apple Maps stops so Apple measures this route rather than substituting its own.
    """
    if n <= 0 or len(nodes) <= 2:
        return []
    indices = [wg.index_of(v) for v in nodes]
    edges = [wg.edge_id(u, v) for u, v in zip(indices, indices[1:])]
    along = np.concatenate([[0.0], np.cumsum(wg.distance[edges])])   # distance at each node
    inner = np.arange(1, len(nodes) - 1)
    picked: List[int] = []
    for target in along[-1] * np.arange(1, n + 1) / (n + 1):
        i = int(inner[np.argmin(np.abs(along[inner] - target))])
        if i not in picked:
            picked.append(i)
    return [int(nodes[i]) for i in sorted(picked)]


def _positive(name: str, values: Sequence[float]) -> List[float]:
    out = [float(v) for v in values]
    if not out or not all(np.isfinite(v) and v > 0 for v in out):
        raise ValueError(f"Apple Maps travel time {name} must be finite and > 0, received {out}")
    return out


def score_trip(
    wg: WeightedGraph,
    legs: Sequence[Sequence[int]],
    free_flow_seconds: np.ndarray,
    gamma: float,
    system: str,
    scenario_type: str,
    apple_eta_optimal: Sequence[float],
    apple_eta_route: Sequence[float],
    trip_id: Optional[str] = None,
) -> dict:
    """
    One system's route for one trip, as a compute_metrics_by_scenario trial.

    `legs` holds the route's node ids per leg (one leg unless multi-destination);
    `apple_eta_optimal` / `apple_eta_route` are Apple's per-leg times for its own route and
    for this one. `gamma` is the system's ETA risk sensitivity.
    """
    if not legs or any(len(leg) < 2 for leg in legs):
        raise ValueError("every leg needs at least two nodes")
    if not (len(apple_eta_optimal) == len(apple_eta_route) == len(legs)):
        raise ValueError(
            f"{len(legs)} leg(s) but {len(apple_eta_optimal)} optimal and {len(apple_eta_route)} route times"
        )
    for a, b in zip(legs, legs[1:]):
        if a[-1] != b[0]:
            raise ValueError(f"leg ending at {a[-1]} does not continue into a leg starting at {b[0]}")
    optimal = _positive("apple_eta_optimal", apple_eta_optimal)
    measured = _positive("apple_eta_route", apple_eta_route)

    path = [int(v) for v in legs[0]] + [int(v) for leg in legs[1:] for v in leg[1:]]
    metrics = wg.evaluate_path(path)
    predicted = [path_eta_seconds(wg, free_flow_seconds, leg, gamma) for leg in legs]
    return {
        "trip_id": trip_id,
        "system": system,
        "scenario_type": scenario_type,
        "window": wg.window,
        "stops": [int(leg[0]) for leg in legs] + [int(legs[-1][-1])],
        "path": path,
        "c_optimal": float(sum(optimal)),
        "c_predicted": float(sum(measured)),
        "actual_eta": measured,
        "predicted_eta": predicted,
        "hops": metrics.n_edges,
        "distance_m": metrics.distance_m,
        "risk_exposure": metrics.risk_exposure,
        "eta_minutes": float(sum(predicted)) / 60.0,
    }
