"""
Route Quality Before and After Condition Updates.

A route is planned under the conditions of one moment; then conditions change -- the next
recorded window's predicted congestion risk, and the incidents in force by then. The question
is whether recalculating under the new conditions gives a better route *under those new
conditions* than keeping the old one.

For one trip and one update, from conditions C_before to C_after:

    old route           ANTROUTE's route planned under C_before
    recalculated route  ANTROUTE's route planned under C_after (same origin and destination)

Both are measured under C_after -- the conditions the driver actually meets:

    congestion exposure   sum of Risk_after(u, v) over the route's edges   (RADR eq. 5)
    event exposure        sum of Event_after(u, v) over the route's edges  (event_layer)
    avoidable event exp.  the same without the edges at the origin and destination, which
                          neither route can avoid
    route length          metres, descriptive only

and how much the route changed: route_changed (any difference) and shared_length_fraction,
the share of the old route's length that the recalculated route still uses (1 = unchanged,
0 = no road in common).

and the improvement is old minus recalculated, so a positive number means recalculating
helped. Paired over trips, the exposures are tested with significance.compare_paired, the
recalculated route in the "proposed" role and the old route in the "baseline" role
(Equation 7: lower is better).

Conditions are the same ones ANTROUTE routes on: Risk from the risk tiers of the whole-city
graph (observed, propagated, assumed) and Event from the Event-Aware layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

from src.routing.dynamic_weight import WeightedGraph
from src.routing.event_layer import event_exposure

METRICS = ("congestion_exposure", "event_exposure", "avoidable_event_exposure")


@dataclass
class RouteUnderConditions:
    distance_m: float
    congestion_exposure: float
    event_exposure: float
    avoidable_event_exposure: float


def measure(wg_after: WeightedGraph, impact_after: Optional[np.ndarray], path: Sequence[int]) -> RouteUnderConditions:
    """A route's exposures under the updated conditions: wg_after's risk, impact_after's incidents."""
    m = wg_after.evaluate_path(path)
    ends = [path[0], path[-1]]
    return RouteUnderConditions(m.distance_m, m.risk_exposure, event_exposure(wg_after, impact_after, path),
                                event_exposure(wg_after, impact_after, path, stops=ends))


def shared_length_fraction(wg: WeightedGraph, old_path: Sequence[int], new_path: Sequence[int]) -> float:
    """Share of the old route's length that the new route also uses (edge by edge)."""
    def edges(path):
        idx = [wg.index_of(int(v)) for v in path]
        return {wg.edge_id(u, v) for u, v in zip(idx, idx[1:])}

    old_edges, new_edges = edges(old_path), edges(new_path)
    old_length = float(wg.distance[list(old_edges)].sum())
    if old_length <= 0:
        return 1.0
    return float(wg.distance[list(old_edges & new_edges)].sum()) / old_length if old_edges & new_edges else 0.0


def compare_update(
    wg_after: WeightedGraph,
    impact_after: Optional[np.ndarray],
    old_path: Sequence[int],
    new_path: Sequence[int],
) -> dict:
    """
    The old route and the recalculated one, both under the updated conditions, and the
    improvement on each metric (old - recalculated; positive = recalculation helped).
    Raises if the two routes do not share an origin and destination.
    """
    if len(old_path) < 2 or len(new_path) < 2:
        raise ValueError("a route needs at least two nodes")
    if (old_path[0], old_path[-1]) != (new_path[0], new_path[-1]):
        raise ValueError("the old and recalculated routes must join the same origin and destination")
    old, new = measure(wg_after, impact_after, old_path), measure(wg_after, impact_after, new_path)
    row = {
        "route_changed": list(map(int, old_path)) != list(map(int, new_path)),
        "shared_length_fraction": shared_length_fraction(wg_after, old_path, new_path),
    }
    for name, value in asdict(old).items():
        row[f"old_{name}"] = value
    for name, value in asdict(new).items():
        row[f"recalculated_{name}"] = value
    for metric in METRICS:
        row[f"{metric}_improvement"] = row[f"old_{metric}"] - row[f"recalculated_{metric}"]
    row["distance_m_change"] = new.distance_m - old.distance_m
    return row
