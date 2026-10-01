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


def _distance_to_destination(wg: WeightedGraph, destination_idx: int) -> np.ndarray:
    import networkx as nx

    g = wg.to_networkx().reverse(copy=False)
    destination_road_id = int(wg.node_ids[destination_idx])
    lengths = nx.single_source_dijkstra_path_length(g, destination_road_id, weight="weight")
    potential = np.full(wg.n_nodes, np.inf)
    for road_id, length in lengths.items():
        potential[wg.index_of(road_id)] = length
    return potential


def _candidate_edges(wg: WeightedGraph, current: int, visited: set, potential: np.ndarray) -> np.ndarray:
    out = wg.out_edges(current)
    if out.size == 0:
        return out
    mask = np.array(
        [int(wg.dst[e]) not in visited and potential[int(wg.dst[e])] < potential[current] for e in out]
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
    desirability = pheromone[candidates] ** alpha * (1.0 / wg.weight[candidates]) ** beta
    total = desirability.sum()
    if total <= 0 or not np.isfinite(total):
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
    potential: np.ndarray,
    rng: np.random.Generator,
) -> Optional[Tuple[List[int], List[int]]]:
    current = origin_idx
    visited = {origin_idx}
    path_indices = [origin_idx]
    edge_ids: List[int] = []
    for _ in range(max_steps):
        if current == destination_idx:
            return (path_indices, edge_ids)
        candidates = _candidate_edges(wg, current, visited, potential)
        if candidates.size == 0:
            return None
        chosen = _choose_edge(wg, candidates, pheromone, config.alpha, config.beta, rng)
        next_node = int(wg.dst[chosen])
        path_indices.append(next_node)
        edge_ids.append(chosen)
        visited.add(next_node)
        current = next_node
    return None


def ant_colony_shortest_path(
    wg: WeightedGraph, origin: int, destination: int, config: AntColonyConfig = AntColonyConfig()
) -> RouteResult:
    origin_idx = wg.index_of(origin)
    destination_idx = wg.index_of(destination)
    if origin_idx == destination_idx:
        raise ValueError(f"origin {origin} and destination {destination} are the same node")
    potential = _distance_to_destination(wg, destination_idx)
    if not np.isfinite(potential[origin_idx]):
        raise ValueError(f"destination {destination} is not reachable from origin {origin}")
    max_steps = config.max_steps or wg.n_nodes
    rng = np.random.default_rng(config.seed)
    pheromone = np.full(wg.n_edges, config.initial_pheromone, dtype=np.float64)
    found: Dict[Tuple[int, ...], PathMetrics] = {}
    for _ in range(config.n_iterations):
        tours: List[Tuple[PathMetrics, List[int]]] = []
        for _ in range(config.n_ants):
            result = _build_tour(
                wg, origin_idx, destination_idx, pheromone, config, max_steps, potential, rng
            )
            if result is None:
                continue
            path_indices, edge_ids = result
            road_ids = [int(wg.node_ids[i]) for i in path_indices]
            metrics = wg.evaluate_path(road_ids)
            tours.append((metrics, edge_ids))
            key = tuple(metrics.nodes)
            if key not in found or metrics.dynamic_cost < found[key].dynamic_cost:
                found[key] = metrics
        pheromone *= 1.0 - config.evaporation
        for metrics, edge_ids in tours:
            pheromone[edge_ids] += 1.0 / metrics.dynamic_cost
    if not found:
        raise ValueError(
            f"no route from {origin} to {destination} was found within {config.n_iterations} iterations x {config.n_ants} ants (max {max_steps} steps per tour)"
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