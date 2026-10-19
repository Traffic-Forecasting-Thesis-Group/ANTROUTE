"""Ant Colony Optimization with a cost heuristic and online convergence rule."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from pydantic import Field

from .dynamic_weight import WeightedRoad
from .network import StudyModel


class AntColonyConfig(StudyModel):
    n_ants: int = Field(default=20, ge=1, le=200, strict=True)
    max_iterations: int = Field(default=60, ge=1, le=1000, strict=True)
    alpha: float = Field(default=1, ge=0, le=5)
    beta: float = Field(default=2, ge=0, le=10)
    evaporation: float = Field(default=0.5, gt=0, lt=1)
    tolerance: float = Field(default=0.0001, ge=0, lt=1)
    patience: int = Field(default=10, ge=1, le=1000, strict=True)
    max_steps: int = Field(default=10000, ge=1, le=100000, strict=True)
    seed: int = Field(default=0, strict=True)
    goal_bias: float = Field(default=4, ge=0, le=10)


class NoFeasibleRoute(ValueError):
    """The directed graph has no feasible route under this snapshot."""


class SearchExhausted(ValueError):
    """ACO did not complete a route within its search limits."""


@dataclass
class ColonyResult:
    roads: list[WeightedRoad]
    best_cost_by_iteration: list[float | None]
    converged: bool
    pheromone: dict[str, float] = field(default_factory=dict)
    best_found_at: int = 0  # Iteration that first found the final best route: iterations to convergence.
    visits: dict[str, int] = field(default_factory=dict)  # Completed ant routes that used each road.
    ant_routes: int = 0  # Completed ant routes over the whole search.


def _reachable(adjacency: dict[str, list[WeightedRoad]], destination: str) -> set[str]:
    reverse: dict[str, list[str]] = {node: [] for node in adjacency}
    for source, roads in adjacency.items():
        for edge in roads:
            reverse[edge.road.target].append(source)
    reachable, pending = {destination}, [destination]
    while pending:
        for source in reverse[pending.pop()]:
            if source not in reachable:
                reachable.add(source)
                pending.append(source)
    return reachable


def ant_colony_route(
    adjacency: dict[str, list[WeightedRoad]], origin: str, destination: str, config: AntColonyConfig,
    positions: dict[str, tuple[float, float]] | None = None, pheromone: dict[str, float] | None = None,
) -> ColonyResult:
    """pheromone: warm start for re-planning; roads without a value start at 1."""
    if origin not in adjacency or destination not in adjacency:
        raise ValueError("Origin and destination must be nodes in the study network")
    if origin == destination:
        raise ValueError("Origin and destination must differ")
    reachable = _reachable(adjacency, destination)
    if origin not in reachable:
        raise NoFeasibleRoute("No feasible route within the study area under these conditions")
    rng = random.Random(config.seed)
    # Straight-line distance is a soft ACO heuristic, not a separate path solver.
    # Every feasible move retains nonzero probability, including necessary detours.
    goal_distance = {}
    if positions is not None:
        longitude, latitude = positions[destination]
        longitude_scale = math.cos(math.radians(latitude))
        goal_distance = {node: math.hypot((point[0] - longitude) * longitude_scale, point[1] - latitude) * 111195
                         for node, point in positions.items()}
    pheromone = {edge.road.id: (pheromone or {}).get(edge.road.id, 1.0) for roads in adjacency.values() for edge in roads}
    best: list[WeightedRoad] | None = None
    best_cost = math.inf
    best_found_at = 0
    history: list[float | None] = []
    stable = 0
    visits: dict[str, int] = {}
    ant_routes = 0
    for iteration in range(1, config.max_iterations + 1):
        previous_cost = best_cost
        completed: list[tuple[list[WeightedRoad], float]] = []
        for _ in range(config.n_ants):
            current = origin
            visited = {origin}
            route: list[WeightedRoad] = []
            # Backtracking gives ants a chance to escape dead ends without revisiting cycles.
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
                # Log weights avoid overflow when configured feature costs are very small.
                scores = [config.alpha * math.log(pheromone[edge.road.id])
                          - config.beta * math.log(max(edge.cost, 1e-12)) for edge in candidates]
                if goal_distance:
                    scores = [score + config.goal_bias * max(-1.0, min(1.0,
                              (goal_distance[current] - goal_distance[edge.road.target]) / max(edge.road.distance_m, 1)))
                              for score, edge in zip(scores, candidates)]
                peak = max(scores)
                chosen = rng.choices(candidates, weights=[math.exp(score - peak) for score in scores], k=1)[0]
                route.append(chosen)
                current = chosen.road.target
                visited.add(current)
            if current == destination:
                cost = sum(edge.cost for edge in route)
                completed.append((route, cost))
                ant_routes += 1
                for edge in route:
                    visits[edge.road.id] = visits.get(edge.road.id, 0) + 1
                if cost < best_cost - 1e-12:
                    best, best_cost, best_found_at = route[:], cost, iteration
        for edge_id in pheromone:
            pheromone[edge_id] = max(pheromone[edge_id] * (1 - config.evaporation), 1e-12)
        for route, cost in completed:
            for edge in route:
                pheromone[edge.road.id] += 1 / max(cost, 1e-12)
        history.append(None if best is None else best_cost)
        if math.isfinite(previous_cost) and best is not None:
            improvement = (previous_cost - best_cost) / max(abs(previous_cost), 1e-12)
            stable = stable + 1 if improvement <= config.tolerance else 0
            if stable >= config.patience:
                return ColonyResult(best, history, True, pheromone, best_found_at, visits, ant_routes)
    if best is None:
        raise SearchExhausted("ACO found no route within its search limits; connectivity is not disproved")
    return ColonyResult(best, history, False, pheromone, best_found_at, visits, ant_routes)
