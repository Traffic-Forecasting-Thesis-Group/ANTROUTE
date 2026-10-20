"""One ACO search kernel for the controlled comparison; each model supplies only its native edge costs.

Transition (Cheng, 2023, eq. 8, 10-11, used by both models):
  eta[e]  = 1 / (native_cost[e] + d(target[e], destination) / distance_reference + epsilon)
  score   = alpha * log(tau[e]) + beta * log(eta[e])                      (log form: no overflow)
  with probability q0 take the best-scoring edge (ties: lowest edge ID), otherwise sample by exp(score).
Pheromone, once per iteration:
  colony stage          tau <- (1 - rho) * tau, then + Q / native_route_cost on each successful ant's edges
  iteration-best stage  tau <- (1 - xi) * tau, then + Q / cost on the iteration-best route's edges
  failed ants deposit nothing; when no ant succeeds only the colony evaporation applies; tau >= floor.
Stopping (cumulative patience window, B[k] = best cost after iteration k):
  converged at k when B[k-p] exists and B[k-p] - B[k] <= abs_tol + rel_tol * |B[k-p]|
  iteration_limit when max_iterations pass first; time_limit when the wall-time budget runs out.
A time limit returns the best route so far when there is one; otherwise SearchTimeLimit. A graph with no
route at all is NoFeasibleRoute, a separate outcome decided before any ant runs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from pydantic import Field

from .network import StudyModel

StopReason = Literal["converged", "iteration_limit", "time_limit"]


class NoFeasibleRoute(ValueError):
    """The directed graph has no feasible route under this snapshot."""


class SearchExhausted(ValueError):
    """ACO did not complete a route within its search limits; connectivity is not disproved."""


class SearchTimeLimit(SearchExhausted):
    """The wall-time budget ran out before any ant completed a route."""


class SearchConfig(StudyModel):
    """Shared search settings of the controlled comparison.

    alpha, rho, xi and the colony size follow Cheng (2023, Sec. 5.1.2). q0, beta and the wall-time budget were frozen
    by scripts/tune_search.py on the tuning period (outputs/search_tuning.json, calibration 34441311a378): the declared
    grid q0 in {0, 0.3, 0.5, 0.7} x beta in {1, 2, 3, 5} chose q0 = 0, beta = 1 (lowest mean optimality gap over both
    models); the budget is 2 x the longest tuning search, rounded up. Cheng's published q0 = 0.7, beta = 5 are not used.
    """
    n_ants: int = Field(default=20, ge=1, le=200, strict=True)
    max_iterations: int = Field(default=60, ge=1, le=1000, strict=True)
    alpha: float = Field(default=1, ge=0, le=5)
    beta: float = Field(default=1, ge=0, le=10)  # Tuned.
    rho: float = Field(default=0.3, gt=0, lt=1)  # Colony-stage evaporation.
    xi: float = Field(default=0.3, ge=0, lt=1)  # Iteration-best-stage evaporation.
    q0: float = Field(default=0.0, ge=0, le=1)  # Exploitation probability; tuned.
    deposit: float = Field(default=1, gt=0)  # Q
    epsilon: float = Field(default=1e-9, gt=0)
    pheromone_floor: float = Field(default=1e-12, gt=0)
    # Convergence rule: patience 20 with zero tolerances reproduces "no improvement for 20 consecutive iterations".
    patience: int = Field(default=20, ge=1, le=1000, strict=True)
    abs_tol: float = Field(default=0, ge=0)
    rel_tol: float = Field(default=0, ge=0, lt=1)
    # Frozen from the tuning period: 2 x the longest tuning search of the chosen setting (8.9 s), rounded up.
    wall_time_ms: float | None = Field(default=18000, gt=0)
    deadline_check_steps: int = Field(default=256, ge=1, strict=True)
    max_steps: int = Field(default=10000, ge=1, le=100000, strict=True)
    seed: int = Field(default=0, strict=True)


@dataclass
class SearchResult:
    roads: list
    best_cost_by_iteration: list[float | None]
    stop_reason: StopReason
    first_best_found_iteration: int  # Iteration that first found the final best route.
    converged_at: int | None  # N_conv: iteration that first met the convergence rule; None if not converged.
    iterations_completed: int
    search_time_ms: float
    pheromone: dict[str, float] = field(default_factory=dict)
    visits: dict[str, int] = field(default_factory=dict)  # Completed ant routes that used each road.
    ant_routes: int = 0  # Completed ant routes over the whole search.

    @property
    def converged(self) -> bool:
        return self.stop_reason == "converged"

    @property
    def best_found_at(self) -> int:
        return self.first_best_found_iteration


def reachable_from(adjacency, destination: str) -> set[str]:
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


def converged(history: list[float | None], patience: int, abs_tol: float, rel_tol: float) -> bool:
    """Cumulative window rule on the best-so-far history: B[k-p] - B[k] <= abs_tol + rel_tol * |B[k-p]|."""
    if len(history) <= patience:
        return False
    before, now = history[-1 - patience], history[-1]
    if before is None or now is None or not math.isfinite(before):
        return False
    return before - now <= abs_tol + rel_tol * abs(before)


def goal_distances(positions, destination) -> dict[str, float]:
    longitude, latitude = positions[destination]
    scale = math.cos(math.radians(latitude))
    return {node: math.hypot((point[0] - longitude) * scale, point[1] - latitude) * 111195 for node, point in positions.items()}


def aco_search(adjacency, origin: str, destination: str, config: SearchConfig, positions, distance_reference_m: float,
               pheromone: dict[str, float] | None = None, started: float | None = None) -> SearchResult:
    """adjacency: node -> outgoing WeightedRoad (cost = the model's native edge cost) under one snapshot.

    pheromone: only for a separately labelled reference mode; controlled searches start fresh (all 1).
    started: perf_counter() when the caller began preparing this search, so preparation counts against the budget.
    """
    started = perf_counter() if started is None else started
    deadline = None if config.wall_time_ms is None else started + config.wall_time_ms / 1000
    if origin not in adjacency or destination not in adjacency:
        raise ValueError("Origin and destination must be nodes in the study network")
    if origin == destination:
        raise ValueError("Origin and destination must differ")
    if not distance_reference_m > 0:
        raise ValueError("The distance reference must be positive")
    reachable = reachable_from(adjacency, destination)
    if origin not in reachable:
        raise NoFeasibleRoute("No feasible route within the study area under these conditions")
    rng = random.Random(config.seed)
    goal = goal_distances(positions, destination)
    # Candidates are sorted by edge ID so ties and sampling order are reproducible; parallel edges keep their IDs.
    ordered = {node: sorted((edge for edge in roads if edge.road.target in reachable), key=lambda edge: edge.road.id)
               for node, roads in adjacency.items()}
    log_eta = {edge.road.id: -math.log(edge.cost + goal[edge.road.target] / distance_reference_m + config.epsilon)
               for roads in ordered.values() for edge in roads}
    tau = {road_id: (pheromone or {}).get(road_id, 1.0) for road_id in log_eta}
    best, best_cost, best_found_at = None, math.inf, 0
    history: list[float | None] = []
    visits: dict[str, int] = {}
    ant_routes = 0

    def finish(reason: StopReason, iterations: int):
        elapsed = (perf_counter() - started) * 1000
        if best is None:
            if reason == "time_limit":
                raise SearchTimeLimit("The search time budget ran out before a route was found; connectivity is not disproved")
            raise SearchExhausted("ACO found no route within its search limits; connectivity is not disproved")
        return SearchResult(best, history, reason, best_found_at, iterations if reason == "converged" else None,
                            iterations, elapsed, tau, visits, ant_routes)

    def late() -> bool:
        return deadline is not None and perf_counter() > deadline

    if late():
        return finish("time_limit", 0)
    for iteration in range(1, config.max_iterations + 1):
        completed = []
        for _ in range(config.n_ants):
            if late():
                return finish("time_limit", iteration - 1)
            current, visited, route, tried = origin, {origin}, [], {}
            for step in range(config.max_steps):
                if current == destination:
                    break
                if step % config.deadline_check_steps == 0 and step and late():
                    return finish("time_limit", iteration - 1)
                candidates = [edge for edge in ordered[current]
                              if edge.road.target not in visited and edge.road.id not in tried.get(current, ())]
                if not candidates:
                    if not route:
                        break
                    # Bounded backtracking: step back from a dead end and never retry that edge from there.
                    visited.remove(current)
                    last = route.pop()
                    current = last.road.source
                    tried.setdefault(current, set()).add(last.road.id)
                    continue
                scores = [config.alpha * math.log(tau[edge.road.id]) + config.beta * log_eta[edge.road.id] for edge in candidates]
                if rng.random() < config.q0:
                    chosen = candidates[max(range(len(scores)), key=scores.__getitem__)]  # First (lowest ID) on ties.
                else:
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
        for road_id in tau:  # Colony stage; failed ants add nothing.
            tau[road_id] *= 1 - config.rho
        for route, cost in completed:
            for edge in route:
                tau[edge.road.id] += config.deposit / max(cost, config.epsilon)
        if completed and config.xi > 0:  # Iteration-best stage.
            route, cost = min(completed, key=lambda item: item[1])
            for road_id in tau:
                tau[road_id] *= 1 - config.xi
            for edge in route:
                tau[edge.road.id] += config.deposit / max(cost, config.epsilon)
        for road_id in tau:
            tau[road_id] = max(tau[road_id], config.pheromone_floor)
        history.append(None if best is None else best_cost)
        if converged(history, config.patience, config.abs_tol, config.rel_tol):
            return finish("converged", iteration)
    return finish("iteration_limit", config.max_iterations)


def exact_shortest_path(adjacency, origin: str, destination: str):
    """Dijkstra reference under the same frozen objective and eligibility mask. Never given to ACO or timed with it."""
    import heapq
    distance, previous, heap = {origin: 0.0}, {}, [(0.0, origin)]
    while heap:
        value, node = heapq.heappop(heap)
        if node == destination:
            break
        if value > distance[node]:
            continue
        for edge in sorted(adjacency[node], key=lambda edge: edge.road.id):
            candidate = value + edge.cost
            if candidate < distance.get(edge.road.target, math.inf):
                distance[edge.road.target], previous[edge.road.target] = candidate, edge
                heapq.heappush(heap, (candidate, edge.road.target))
    if destination not in previous:
        return None
    route, node = [], destination
    while node != origin:
        route.append(previous[node])
        node = previous[node].road.source
    return route[::-1]
