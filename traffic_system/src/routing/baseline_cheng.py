"""Baseline: the improved ant colony algorithm of Cheng (2023).

J. Cheng, "Dynamic Path Optimization Based on Improved Ant Colony Algorithm",
Journal of Advanced Transportation, 2023, 7651100. https://doi.org/10.1155/2023/7651100

Runs on the same road graph, closures, date/hour traffic and weather as ANTROUTE, but
routes with the paper's own cost and search rules:
  - Eq. 5-6: s_ij = w1 * length + w2 * travel time + w3 * traffic flow, each term divided
    by its maximum over the open roads; w1-w3 from the AHP judgment matrix of Table 3.
  - Eq. 8: heuristic eta_ij = 1 / (s_ij + d_jD), d_jD the straight-line distance from j to D.
  - Eq. 10-11: with probability q0 take the best edge, otherwise sample by tau^alpha * eta^beta.
  - Eq. 12-14: local update from every ant's tour, global update from the iteration's best tour.
  - Step 7: a fixed NC_max iterations, with no early stop.

Inputs and adaptations, stated so results are not over-read:
  - Traffic flow: the study has no vehicle counts. Flow is estimated as demand-to-capacity
    x times road capacity, with x recovered from the congestion index by inverting the BPR
    function that produced it. It is an estimate, like the congestion it comes from.
  - d_jD is divided by the longest road length, the divisor of the length term, so d_ij and
    d_jD share one scale as the A*-style sum in eq. 7 requires. The paper does not give it.
  - xi (global evaporation) is not given in the paper; 0.3, equal to rho.
  - Ants: the paper uses 1.5 x nodes (18,597 here); 20, ANTROUTE's default, keeps run time comparable.
  - Dead ends: as in ANTROUTE, ants only enter nodes that can still reach D and step back from
    dead ends. The paper's test networks have none; under the literal tabu rule ants die on real roads.
  - Eq. 9 (pheromone rescaling when road information changes) runs in simulated trips
    (dynamic_trip.py), where the hour, weather or closures change at a node. It is applied as
    printed, so a road that becomes slower gains pheromone. A single /routes/baseline request is
    one date/hour snapshot, where it never triggers. Traffic updates hourly here, not every
    5-15 minutes as in the paper.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from pydantic import Field

from .aco_routing import NoFeasibleRoute, SearchExhausted, _reachable
from .condition_update import Conditions, validate_conditions
from .dynamic_weight import WeightedRoad, segment_values
from .network import RoadNetwork, StudyModel, WeatherEffect

REFERENCE = {"name": "Improved ant colony algorithm (Cheng, 2023)", "doi": "10.1155/2023/7651100",
             "citation": "J. Cheng, Dynamic Path Optimization Based on Improved Ant Colony Algorithm, "
                         "Journal of Advanced Transportation, 2023, 7651100."}
# Table 3: path length, travel time and traffic flow compared on Saaty's scale (Table 1).
JUDGMENT = ((1, 3, 5), (1 / 3, 1, 3), (1 / 5, 1 / 3, 1))
# Table 2: average random consistency index by matrix order.
RANDOM_INDEX = (0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)


class ChengConfig(StudyModel):
    n_ants: int = Field(default=20, ge=1, le=200, strict=True)
    max_iterations: int = Field(default=60, ge=1, le=1000, strict=True)
    alpha: float = Field(default=1, ge=0, le=5)  # Sec. 5.1.2 (3)
    beta: float = Field(default=5, ge=0, le=10)  # Sec. 5.1.2 (4)
    rho: float = Field(default=0.3, gt=0, lt=1)  # Sec. 5.1.2 (5)
    xi: float = Field(default=0.3, gt=0, lt=1)
    q0: float = Field(default=0.7, ge=0, le=1)  # Large network, Sec. 4.2.1
    pheromone_constant: float = Field(default=1, gt=0)
    max_steps: int = Field(default=10000, ge=1, le=100000, strict=True)
    seed: int = Field(default=0, strict=True)


@dataclass(frozen=True)
class Ahp:
    weights: tuple[float, ...]
    lambda_max: float
    consistency_ratio: float


@dataclass
class BaselineResult:
    roads: list[WeightedRoad]
    best_cost_by_iteration: list[float | None]
    best_found_at: int
    pheromone: dict[str, float] = field(default_factory=dict)


def ahp(matrix=JUDGMENT) -> Ahp:
    """Sec. 3.2.3 steps 1-4 (row geometric mean) and the consistency ratio of eq. 4."""
    n = len(matrix)
    roots = [math.prod(row) ** (1 / n) for row in matrix]
    weights = [root / sum(roots) for root in roots]
    lambda_max = sum(sum(a * w for a, w in zip(row, weights)) / (n * weight) for row, weight in zip(matrix, weights))
    ratio = (lambda_max - n) / (n - 1) / RANDOM_INDEX[n - 1] if n > 2 else 0.0
    return Ahp(tuple(weights), lambda_max, ratio)


def estimated_flow_vph(congestion: float, capacity_vph: float, bpr_alpha: float, bpr_beta: float) -> float:
    """Invert congestion = 1 - 1 / (1 + a * x^b) for demand-to-capacity x; flow = x * capacity."""
    congestion = min(congestion, 1 - 1e-9)
    if congestion <= 0:
        return 0.0
    return ((1 / (1 - congestion) - 1) / bpr_alpha) ** (1 / bpr_beta) * capacity_vph


def comprehensive_costs(criteria: list[tuple[float, float, float]], weights=None) -> list[float]:
    """Eq. 5-6 for (length, travel time, flow) rows: each column divided by its maximum."""
    weights = weights or ahp().weights
    peaks = [max((row[i] for row in criteria), default=0) or 1 for i in range(3)]
    return [sum(w * value / peak for w, value, peak in zip(weights, row, peaks)) for row in criteria]


def cheng_graph(network: RoadNetwork, conditions: Conditions, weather_by_city: dict[str, WeatherEffect] | None = None,
                traffic: dict[str, tuple[float, float]] | None = None) -> dict[str, list[WeightedRoad]]:
    """Open roads weighted by the comprehensive cost s_ij instead of the ten-feature cost."""
    return cheng_snapshot(network, conditions, weather_by_city, traffic)[0]


def cheng_snapshot(network: RoadNetwork, conditions: Conditions, weather_by_city: dict[str, WeatherEffect] | None = None,
                   traffic: dict[str, tuple[float, float]] | None = None
                   ) -> tuple[dict[str, list[WeightedRoad]], dict[str, float]]:
    """The cost graph and each open road's travel time (s), which eq. 9 compares between snapshots."""
    validate_conditions(network, conditions)
    assumptions = network.provenance.get("assumptions", {})
    bpr_alpha, bpr_beta = assumptions.get("bpr_alpha", 0.15), assumptions.get("bpr_beta", 4.0)
    nodes = {node.id: node for node in network.nodes}
    roads, criteria = [], []
    for road in network.roads:
        if road.id in conditions.closed_road_ids:
            continue
        raw, _, _ = segment_values(network, road, conditions, nodes, weather_by_city, traffic)
        roads.append(road)
        criteria.append((raw["distance"], raw["travel_time"],
                         estimated_flow_vph(raw["traffic_congestion"], raw["road_capacity"], bpr_alpha, bpr_beta)))
    adjacency: dict[str, list[WeightedRoad]] = {node.id: [] for node in network.nodes}
    for road, cost in zip(roads, comprehensive_costs(criteria)):
        adjacency[road.source].append(WeightedRoad(road, cost))
    return adjacency, {road.id: row[1] for road, row in zip(roads, criteria)}


def rescale_pheromone(pheromone: dict[str, float], old_time: dict[str, float], new_time: dict[str, float],
                      initial: float = 1.0) -> list[str]:
    """Eq. 9 as written: tau(t+1) = t_(t+1) / t_t * tau(t). Returns the roads whose time changed.

    As printed, a road that becomes slower gains pheromone. Closed roads lose their pheromone and
    reopened roads start again from the initial value.
    """
    changed = []
    for road_id in list(pheromone):
        if road_id not in new_time:
            del pheromone[road_id]
            changed.append(road_id)
        elif abs(new_time[road_id] - old_time[road_id]) > 1e-9 * old_time[road_id]:
            pheromone[road_id] *= new_time[road_id] / old_time[road_id]
            changed.append(road_id)
    for road_id in new_time.keys() - old_time.keys():
        pheromone[road_id] = initial
        changed.append(road_id)
    return changed


def cheng_route(adjacency: dict[str, list[WeightedRoad]], origin: str, destination: str, config: ChengConfig,
                positions: dict[str, tuple[float, float]], pheromone: dict[str, float] | None = None) -> BaselineResult:
    """pheromone: the colony's pheromone after eq. 9, for a re-search; roads without a value start at 1."""
    if origin not in adjacency or destination not in adjacency:
        raise ValueError("Origin and destination must be nodes in the study network")
    if origin == destination:
        raise ValueError("Origin and destination must differ")
    reachable = _reachable(adjacency, destination)
    if origin not in reachable:
        raise NoFeasibleRoute("No feasible route within the study area under these conditions")
    rng = random.Random(config.seed)
    unit = max((edge.road.distance_m for roads in adjacency.values() for edge in roads), default=1)
    longitude, latitude = positions[destination]
    longitude_scale = math.cos(math.radians(latitude))
    goal = {node: math.hypot((point[0] - longitude) * longitude_scale, point[1] - latitude) * 111195 / unit
            for node, point in positions.items()}
    # Eq. 8 depends only on the edge and D, so eta^beta is fixed for the search.
    eta_beta = {edge.road.id: (1 / max(edge.cost + goal[edge.road.target], 1e-12)) ** config.beta
                for roads in adjacency.values() for edge in roads}
    pheromone = {road_id: (pheromone or {}).get(road_id, 1.0) for road_id in eta_beta}
    best: list[WeightedRoad] | None = None
    best_cost, best_found_at = math.inf, 0
    history: list[float | None] = []
    for iteration in range(1, config.max_iterations + 1):
        completed: list[tuple[list[WeightedRoad], float]] = []
        for _ in range(config.n_ants):
            current = origin
            visited = {origin}
            route: list[WeightedRoad] = []
            tried: dict[str, set[str]] = {}
            for _ in range(config.max_steps):
                if current == destination:
                    break
                candidates = [edge for edge in adjacency[current]
                              if edge.road.target in reachable and edge.road.target not in visited
                              and edge.road.id not in tried.get(current, set())]
                if not candidates:
                    if not route:
                        break
                    visited.remove(current)
                    last = route.pop()
                    current = last.road.source
                    tried.setdefault(current, set()).add(last.road.id)
                    continue
                scores = [pheromone[edge.road.id] ** config.alpha * eta_beta[edge.road.id] for edge in candidates]
                # Eq. 10: exploit the best edge with probability q0, otherwise eq. 11.
                if rng.random() < config.q0:
                    chosen = candidates[max(range(len(scores)), key=scores.__getitem__)]
                elif sum(scores) > 0:
                    chosen = rng.choices(candidates, weights=scores, k=1)[0]
                else:
                    chosen = rng.choice(candidates)
                route.append(chosen)
                current = chosen.road.target
                visited.add(current)
            if current == destination:
                completed.append((route, sum(edge.cost for edge in route)))
        # Eq. 12-13: local update from every completed tour.
        for edge_id in pheromone:
            pheromone[edge_id] *= 1 - config.rho
        for route, cost in completed:
            for edge in route:
                pheromone[edge.road.id] += config.pheromone_constant / max(cost, 1e-12)
        if completed:
            route, cost = min(completed, key=lambda tour: tour[1])
            # Eq. 14: global update from this iteration's best tour.
            for edge_id in pheromone:
                pheromone[edge_id] *= 1 - config.xi
            for edge in route:
                pheromone[edge.road.id] += config.pheromone_constant / max(cost, 1e-12)
            if cost < best_cost - 1e-12:
                best, best_cost, best_found_at = route[:], cost, iteration
        history.append(None if best is None else best_cost)
    if best is None:
        raise SearchExhausted("The baseline found no route within its search limits; connectivity is not disproved")
    return BaselineResult(best, history, best_found_at, pheromone)
