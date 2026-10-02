
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from src.routing.dynamic_weight import PathMetrics, WeightedGraph


@dataclass
class AntColonyConfig:
    n_ants: int = 20
    n_iterations: int = 60
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.5
    initial_pheromone: float = 1.0
    max_steps: Optional[int] = None
    n_alternatives: int = 3
    seed: Optional[int] = None


@dataclass
class RouteResult:
    best: PathMetrics
    alternatives: List[PathMetrics]


def _candidate_edges(wg: WeightedGraph, current: int, visited: set) -> np.ndarray:
    out = wg.out_edges(current)
    if out.size == 0:
        return out
    mask = np.array(
        [int(wg.dst[e]) not in visited for e in out],
        dtype=bool,
    )
    return out[mask]


def _choose_edge(
    wg: WeightedGraph,
    candidates: np.ndarray,
    pheromone: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> int:
    weights = wg.weight[candidates]
    if (
        np.any(~np.isfinite(weights))
        or np.any(weights <= 0)
        or np.any(~np.isfinite(pheromone[candidates]))
        or np.any(pheromone[candidates] < 0)
    ):
        raise ValueError(
            "Candidate edge weights must be finite and positive, "
            "and pheromone values must be finite and non-negative."
        )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        desirability = pheromone[candidates] ** alpha * (1.0 / weights) ** beta

    total = desirability.sum()
    if (
        not np.all(np.isfinite(desirability))
        or not np.isfinite(total)
        or total <= 0
    ):
        inverse_weights = 1.0 / weights
        inverse_total = inverse_weights.sum()
        if np.isfinite(inverse_total) and inverse_total > 0:
            probabilities = inverse_weights / inverse_total
        else:
            probabilities = np.full(len(candidates), 1.0 / len(candidates))
    else:
        probabilities = desirability / total

    return int(rng.choice(candidates, p=probabilities))


def _build_tour(
    wg: WeightedGraph,
    origin_idx: int,
    destination_idx: int,
    pheromone: np.ndarray,
    config: AntColonyConfig,
    max_steps: int,
    rng: np.random.Generator,
) -> Optional[Tuple[List[int], List[int]]]:
    current = origin_idx
    visited = {origin_idx}
    path_indices = [origin_idx]
    edge_ids: List[int] = []
    for _ in range(max_steps):
        if current == destination_idx:
            return (path_indices, edge_ids)
        candidates = _candidate_edges(wg, current, visited)
        if candidates.size == 0:
            return None
        chosen = _choose_edge(wg, candidates, pheromone, config.alpha, config.beta, rng)
        next_node = int(wg.dst[chosen])
        path_indices.append(next_node)
        edge_ids.append(chosen)
        visited.add(next_node)
        current = next_node
    if current == destination_idx:
        return (path_indices, edge_ids)
    return None


def ant_colony_shortest_path(
    wg: WeightedGraph, origin: int, destination: int, config: AntColonyConfig = AntColonyConfig()
) -> RouteResult:
    origin_idx = wg.index_of(origin)
    destination_idx = wg.index_of(destination)
    if origin_idx == destination_idx:
        raise ValueError(f"origin {origin} and destination {destination} are the same node")
    if config.n_ants <= 0:
        raise ValueError("n_ants must be greater than zero")
    if config.n_iterations <= 0:
        raise ValueError("n_iterations must be greater than zero")
    if config.alpha < 0 or config.beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    if not 0.0 <= config.evaporation <= 1.0:
        raise ValueError("evaporation must be between 0 and 1")
    if not np.isfinite(config.initial_pheromone) or config.initial_pheromone <= 0:
        raise ValueError("initial_pheromone must be finite and positive")
    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError("max_steps must be greater than zero")
    if config.n_alternatives < 0:
        raise ValueError("n_alternatives must be non-negative")

    max_steps = config.max_steps if config.max_steps is not None else wg.n_nodes - 1
    rng = np.random.default_rng(config.seed)
    pheromone = np.full(wg.n_edges, config.initial_pheromone, dtype=np.float64)
    found: Dict[Tuple[int, ...], PathMetrics] = {}
    for _ in range(config.n_iterations):
        tours: List[Tuple[PathMetrics, List[int]]] = []
        for _ in range(config.n_ants):
            result = _build_tour(
                wg, origin_idx, destination_idx, pheromone, config, max_steps, rng
            )
            if result is None:
                continue
            path_indices, edge_ids = result
            road_ids = [int(wg.node_ids[i]) for i in path_indices]
            metrics = wg.evaluate_path(road_ids)
            if not np.isfinite(metrics.dynamic_cost) or metrics.dynamic_cost <= 0:
                raise ValueError(
                    "Route dynamic_cost must be finite and positive "
                    "for pheromone updates."
                )
            tours.append((metrics, edge_ids))
            key = tuple(metrics.nodes)
            if key not in found or metrics.dynamic_cost < found[key].dynamic_cost:
                found[key] = metrics
        pheromone *= 1.0 - config.evaporation
        for metrics, edge_ids in tours:
            if edge_ids:
                pheromone[edge_ids] += 1.0 / metrics.dynamic_cost
    if not found:
        raise ValueError(
            f"No route from {origin} to {destination} was found "
            f"after {config.n_iterations} iterations with "
            f"{config.n_ants} ants and a maximum of "
            f"{max_steps} steps per tour."
        )
    ranked = sorted(found.values(), key=lambda m: m.dynamic_cost)
    best = ranked[0]
    alternatives = ranked[1 : 1 + config.n_alternatives]
    return RouteResult(best=best, alternatives=alternatives)


def multi_stop_route(
    wg: WeightedGraph, stops: Sequence[int], config: AntColonyConfig = AntColonyConfig()
) -> RouteResult:
    if len(stops) < 2:
        raise ValueError("multi_stop_route needs at least an origin and a destination")
    full_path: List[int] = [int(stops[0])]
    for leg_origin, leg_destination in zip(stops, stops[1:]):
        leg = ant_colony_shortest_path(wg, leg_origin, leg_destination, config)
        full_path.extend(leg.best.nodes[1:])
    best = wg.evaluate_path(full_path)
    return RouteResult(best=best, alternatives=[])