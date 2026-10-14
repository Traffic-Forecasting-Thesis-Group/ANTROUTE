"""
Turn routes and their Google Maps travel times into the trial records src.routing.metrics scores.

Scenarios follow the thesis (Section 3.8, Appendices 1-3):

    recommended        the best route ACO returns for an origin/destination
    alternative        the next-best route from the same ACO run
    multi_destination  a sequential multi-stop trip, routed leg by leg
    incident_exposed   like recommended, but origin and destination are chosen so that a
                       reported incident lies on the INTERIOR of the risk-only route

The fourth exists because the first three cannot measure the road-event conditions of RQ2.
They draw origins and destinations from the CCTV intersections, and the Event-Aware layer
matches incidents to those same intersections through the shared gazetteer, so an incident
lands on the trip's own endpoint: over 120 legs it did so on 8 of the 10 that met one, and
no router can avoid its own destination. Reported as its own scenario and never merged into
the others: selecting trips that can exercise a mechanism is what makes an ablation
informative, but pooling them with trips drawn uniformly would overstate how often the
mechanism bites in ordinary use.

Google Maps is the reference. Route Optimality (Equation 1) uses travel time as the cost,
with Google Maps timing both sides, so it measures route quality and nothing else:

    C_optimal    Google Maps time of Google's own route (summed over legs)
    C_predicted  Google Maps time of the route the system chose (entered via waypoints)

Beside it, measured on the road graph under the window's conditions:

    congestion_exposure        sum of predicted Risk(u, v) over the route (RADR eq. 5)
    event_exposure             sum of Event(u, v), the Event-Aware layer's incident impact
    avoidable_event_exposure   the same, leaving out edges at the trip's own stops, which no
                               route can avoid -- what routing could actually have avoided
    distance_m           route length, descriptive only

ETA is not a research question or metric here: no system's own travel-time estimate is scored.

Two systems are scored the same way:

    antroute   ACO on the risk-weighted graph (lambda > 0) with the Event-Aware layer (mu > 0)
    baseline   Improved ACO (Cheng 2023) on observed camera congestion (see evaluate_routing.py)
"""

from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
from src.routing.dynamic_weight import WeightedGraph
from src.routing.event_layer import event_exposure

ANTROUTE = "antroute"
BASELINE = "baseline"
SYSTEMS = (ANTROUTE, BASELINE)

RECOMMENDED = "recommended"
ALTERNATIVE = "alternative"
MULTI_DESTINATION = "multi_destination"
INCIDENT_EXPOSED = "incident_exposed"
SCENARIOS = (RECOMMENDED, ALTERNATIVE, MULTI_DESTINATION, INCIDENT_EXPOSED)

# How each Google Maps time was obtained. Every scored row of a lookup sheet records all four,
# and metrics.json summarises them (scripts/evaluate_routing.py reference_record).
REFERENCE_FIELDS = ("google_access_method", "google_departure_date", "google_queried_on", "google_traffic_model")


def pick_waypoints(wg: WeightedGraph, nodes: Sequence[int], n: int) -> List[int]:
    """
    Up to `n` intermediate nodes spaced evenly by distance along the route, to enter as
    Google Maps waypoints so Google measures this route rather than substituting its own.
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
        raise ValueError(f"Google Maps travel time {name} must be finite and > 0, received {out}")
    return out


def score_trip(
    wg: WeightedGraph,
    legs: Sequence[Sequence[int]],
    system: str,
    scenario_type: str,
    reference_optimal: Sequence[float],
    reference_route: Sequence[float],
    trip_id: Optional[str] = None,
    impact: Optional[np.ndarray] = None,
) -> dict:
    """
    One system's route for one trip, as a compute_metrics_by_scenario trial.

    `legs` holds the route's node ids per leg (one leg unless multi-destination);
    `reference_optimal` / `reference_route` are Google Maps' per-leg times for its own route
    and for this one. `wg` measures distance and congestion risk, and `impact` ([E], aligned
    to `wg`) the incidents in force, for every system alike -- one yardstick for both.
    """
    if not legs or any(len(leg) < 2 for leg in legs):
        raise ValueError("every leg needs at least two nodes")
    if not (len(reference_optimal) == len(reference_route) == len(legs)):
        raise ValueError(
            f"{len(legs)} leg(s) but {len(reference_optimal)} optimal and {len(reference_route)} route times"
        )
    for a, b in zip(legs, legs[1:]):
        if a[-1] != b[0]:
            raise ValueError(f"leg ending at {a[-1]} does not continue into a leg starting at {b[0]}")
    optimal = _positive("reference_optimal", reference_optimal)
    measured = _positive("reference_route", reference_route)

    path = [int(v) for v in legs[0]] + [int(v) for leg in legs[1:] for v in leg[1:]]
    metrics = wg.evaluate_path(path, impact)
    return {
        "trip_id": trip_id,
        "system": system,
        "scenario_type": scenario_type,
        "window": wg.window,
        "stops": [int(leg[0]) for leg in legs] + [int(legs[-1][-1])],
        "path": path,
        "c_optimal": float(sum(optimal)),
        "c_predicted": float(sum(measured)),
        "reference_route_seconds": measured,
        "hops": metrics.n_edges,
        "distance_m": metrics.distance_m,
        "congestion_exposure": metrics.risk_exposure,
        "event_exposure": metrics.event_exposure,
        "avoidable_event_exposure": event_exposure(
            wg, impact, path, stops=[int(leg[0]) for leg in legs] + [int(legs[-1][-1])]
        ),
    }
