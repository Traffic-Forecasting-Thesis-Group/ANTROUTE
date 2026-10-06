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
Apple's time for that same route, per leg, as Appendix 2's note specifies. Given the Apple
Maps time at each waypoint, they are scored per segment instead (stop to stop along the
route), so every trial has several ETA observations and a per-trial R^2 -- a single leg has
one observation, for which R^2 is undefined.

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


def split_at(nodes: Sequence[int], waypoints: Sequence[int]) -> List[List[int]]:
    """The route cut at its Apple Maps stops: one node list per stop-to-stop segment."""
    nodes = [int(v) for v in nodes]
    segments, start = [], 0
    for w in waypoints:
        try:
            i = nodes.index(int(w), start + 1, len(nodes) - 1)
        except ValueError:
            raise ValueError(f"waypoint {w} is not an inner node of the route after {nodes[start]}") from None
        segments.append(nodes[start:i + 1])
        start = i
    segments.append(nodes[start:])
    return segments


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
    predicted_eta: Optional[Sequence[float]] = None,
    waypoints: Optional[Sequence[Sequence[int]]] = None,
    apple_eta_segments: Optional[Sequence[Sequence[float]]] = None,
) -> dict:
    """
    One system's route for one trip, as a compute_metrics_by_scenario trial.

    `legs` holds the route's node ids per leg (one leg unless multi-destination);
    `apple_eta_optimal` / `apple_eta_route` are Apple's per-leg times for its own route and
    for this one. `gamma` is the system's ETA risk sensitivity.

    `predicted_eta` supplies the system's own per-leg travel-time estimate when it does not
    come from ANTROUTE's ETA engine -- the baseline (Improved ACO) predicts with the observed
    travel times its own cost function is built from. Left out, the estimate is computed from
    `wg` and `gamma` as before. Note that `wg` still measures distance and risk for every
    system, so those stay on one yardstick.

    `waypoints` (the Apple Maps stops per leg) with `apple_eta_segments` (Apple's time for
    each stop-to-stop segment, per leg) scores the ETA per segment: `actual_eta` and
    `predicted_eta` then hold one value per segment, and `predicted_eta`, if given, must too.
    Route Optimality still uses the per-leg totals.
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

    # The pieces the ETA is scored over: whole legs, or each leg's stop-to-stop segments.
    pieces, actual = legs, measured
    if apple_eta_segments is not None:
        if waypoints is None or len(waypoints) != len(legs) or len(apple_eta_segments) != len(legs):
            raise ValueError("segment times need the waypoints and the segment times of every leg")
        pieces, actual = [], []
        for leg, stops, times in zip(legs, waypoints, apple_eta_segments):
            segments = split_at(leg, stops)
            if len(times) != len(segments):
                raise ValueError(f"leg with {len(segments)} segment(s) but {len(times)} segment time(s)")
            pieces += segments
            actual += _positive("apple_eta_segments", times)

    if predicted_eta is None:
        predicted = [path_eta_seconds(wg, free_flow_seconds, piece, gamma) for piece in pieces]
    else:
        predicted = _positive("predicted_eta", predicted_eta)
        if len(predicted) != len(pieces):
            raise ValueError(f"{len(pieces)} leg(s)/segment(s) but {len(predicted)} predicted travel time(s)")
    return {
        "trip_id": trip_id,
        "system": system,
        "scenario_type": scenario_type,
        "window": wg.window,
        "stops": [int(leg[0]) for leg in legs] + [int(legs[-1][-1])],
        "path": path,
        "c_optimal": float(sum(optimal)),
        "c_predicted": float(sum(measured)),
        "actual_eta": actual,
        "predicted_eta": predicted,
        "eta_points": len(actual),
        "hops": metrics.n_edges,
        "distance_m": metrics.distance_m,
        "risk_exposure": metrics.risk_exposure,
        "eta_minutes": float(sum(predicted)) / 60.0,
    }
