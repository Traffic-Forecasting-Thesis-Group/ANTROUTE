"""
Ant Colony Optimization (ACO) routing over a risk-weighted road graph.

Edge costs come from dynamic_weight.WeightedGraph (free-flow travel time scaled by the
predicted Congestion Risk Score). Each iteration, every ant walks origin -> destination,
choosing edges with probability ~ pheromone^alpha * (1 / (1 + regret))^beta, where regret is
how much worse the edge is than the best option at that junction under an exact remaining-cost
heuristic (see remaining_cost_to). Pheromone then evaporates and each completed tour deposits
1 / route_cost on its edges. diverse_routes layers a penalty method on top for alternatives.
"""

from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from src.routing.dynamic_weight import PathMetrics, WeightedGraph


@dataclass
class AntColonyConfig:
    n_ants: int = 20                  # tours walked per iteration
    n_iterations: int = 60            # pheromone update rounds
    alpha: float = 1.0                # weight of pheromone in the edge choice
    beta: float = 2.0                 # weight of the cost heuristic (regret) in the edge choice
    evaporation: float = 0.5          # fraction of pheromone lost each iteration
    initial_pheromone: float = 1.0
    max_steps: Optional[int] = None   # per-tour step cap; None = number of nodes - 1
    n_alternatives: int = 3           # extra distinct tours returned besides the best
    seed: Optional[int] = None        # RNG seed, for reproducible routes


@dataclass
class RouteResult:
    """Cheapest route found, plus the next-cheapest distinct tours the ants walked."""

    best: PathMetrics
    alternatives: List[PathMetrics]


def remaining_cost_to(wg: WeightedGraph, destination_idx: int) -> np.ndarray:
    """
    Shortest remaining dynamic_cost from every node to `destination_idx`, via one Dijkstra run
    on the reversed graph (so distances are "to", not "from", the destination).

    This is what turns the ant colony into a goal-directed search: with only local edge cost
    to go on, an ant choosing among 2-3 outgoing edges at each node has no sense of which way
    the destination actually is, so on a multi-hop route (tens of hops is normal at city scale)
    it essentially never completes a tour by chance, and pheromone never gets a real route to
    reinforce. Biasing each choice by the estimated remaining cost (an A* heuristic bolted onto
    ACO) fixes that while keeping the pheromone/randomness that gives alternative routes.
    """
    n = wg.n_nodes
    reversed_graph = sp.csr_matrix((wg.weight, (wg.dst, wg.src)), shape=(n, n))
    distances = dijkstra(reversed_graph, directed=True, indices=destination_idx)
    return distances


def _choose_edge(
    wg: WeightedGraph,
    candidates: np.ndarray,
    pheromone: np.ndarray,
    remaining: np.ndarray,
    current_remaining: float,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> int:
    # weight+remaining[dst] is the estimated total tour cost through this candidate, which by
    # the triangle inequality is always >= current_remaining (equality iff the edge lies on a
    # shortest path). That total is tens of thousands at city scale while the gap between a
    # good and a bad candidate is usually a few hundred -- a couple percent -- so inverting the
    # raw total barely discriminates between candidates even at beta=2, and over a 60+ hop
    # route the odds of guessing right every single step collapse to ~0 (confirmed: 0/500
    # single-ant tours completed when desirability was based on the raw total). Using the
    # regret -- how much worse this edge is than the best option at this junction, a small
    # number -- instead of the raw total restores real discriminating power.
    weights = wg.weight[candidates] + remaining[wg.dst[candidates]]
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
    regret = np.maximum(weights - current_remaining, 0.0)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        desirability = pheromone[candidates] ** alpha * (1.0 / (regret + 1.0)) ** beta

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

    # rng.choice(candidates, p=probabilities) step for step -- same cumulative distribution,
    # same single uniform draw -- without its per-call validation, which made this one line
    # most of the routing time.
    cdf = np.cumsum(probabilities)
    cdf /= cdf[-1]
    return int(candidates[int(np.searchsorted(cdf, rng.random(), side="right"))])


# Road layout -> adjacency lists. Graphs that differ only in weight (diverse_routes'
# penalised copies, other risk windows) share src/dst arrays, so they share this.
_ADJACENCY_CACHE: List[Tuple[np.ndarray, np.ndarray, List[List[Tuple[int, int]]]]] = []


def _adjacency(wg: WeightedGraph) -> List[List[Tuple[int, int]]]:
    """(edge, head) for the edges out of each node, in out_edges order, as plain ints."""
    for src, dst, adjacency in _ADJACENCY_CACHE:
        if src is wg.src and dst is wg.dst:
            return adjacency
    edges = wg._out_edges.tolist()
    heads = wg.dst[wg._out_edges].tolist()
    starts = wg._out_start.tolist()
    adjacency = [list(zip(edges[a:b], heads[a:b])) for a, b in zip(starts, starts[1:])]
    _ADJACENCY_CACHE.insert(0, (wg.src, wg.dst, adjacency))
    del _ADJACENCY_CACHE[4:]
    return adjacency


@dataclass
class _ColonyInputs:
    """What every ant of one colony reads, as plain Python values for the inner loop."""

    adjacency: List[List[Tuple[int, int]]]
    reachable: List[bool]       # [N] the destination can be reached from this node
    estimate: List[float]       # [E] weight + remaining cost at the edge's head
    remaining: List[float]      # [N]

    @classmethod
    def build(cls, wg: WeightedGraph, remaining: np.ndarray) -> "_ColonyInputs":
        return cls(
            adjacency=_adjacency(wg),
            reachable=np.isfinite(remaining).tolist(),
            estimate=(wg.weight + remaining[wg.dst]).tolist(),
            remaining=remaining.tolist(),
        )


def _choose_scored(
    candidates: List[int],
    estimate: List[float],
    pheromone: List[float],
    current_remaining: float,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    wg: WeightedGraph,
    pheromone_array: np.ndarray,
    remaining: np.ndarray,
) -> int:
    """
    _choose_edge on plain floats, for the inner loop: the same validation, regret,
    desirability, cumulative distribution and single uniform draw. Anything non-finite
    goes to _choose_edge itself, which owns the fallbacks.
    """
    weights = [estimate[e] for e in candidates]
    tau = [pheromone[e] for e in candidates]
    if not all(0.0 < w < np.inf for w in weights) or not all(0.0 <= t < np.inf for t in tau):
        return _choose_edge(
            wg, np.array(candidates, dtype=np.int64), pheromone_array, remaining,
            current_remaining, alpha, beta, rng,
        )
    desirability = [t ** alpha * (1.0 / (max(w - current_remaining, 0.0) + 1.0)) ** beta for t, w in zip(tau, weights)]
    total = float(np.array(desirability).sum())
    if not (0.0 < total < np.inf):
        return _choose_edge(
            wg, np.array(candidates, dtype=np.int64), pheromone_array, remaining,
            current_remaining, alpha, beta, rng,
        )
    cdf, running = [], 0.0
    for d in desirability:
        running += d / total
        cdf.append(running)
    last = cdf[-1]
    cdf = [c / last for c in cdf]
    return candidates[bisect_right(cdf, rng.random())]


def _build_tour(
    wg: WeightedGraph,
    origin_idx: int,
    destination_idx: int,
    pheromone: np.ndarray,
    remaining: np.ndarray,
    config: AntColonyConfig,
    max_steps: int,
    rng: np.random.Generator,
    colony: Optional["_ColonyInputs"] = None,
    pheromone_list: Optional[List[float]] = None,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    One ant's tour. `colony` and `pheromone_list` (pheromone as floats) are the inputs
    the colony precomputes once; without them they are built here.
    """
    if colony is None:
        colony = _ColonyInputs.build(wg, remaining)
    if pheromone_list is None:
        pheromone_list = pheromone.tolist()
    adjacency, reachable, estimate, remaining_list = (
        colony.adjacency, colony.reachable, colony.estimate, colony.remaining
    )
    visited = bytearray(wg.n_nodes)
    visited[origin_idx] = 1
    path_indices = [origin_idx]
    edge_ids: List[int] = []
    blocked_from: Dict[int, set] = {}
    steps = 0
    while steps < max_steps:
        current = path_indices[-1]
        if current == destination_idx:
            return (path_indices, edge_ids)
        blocked = blocked_from.get(current, ())
        candidates = [
            e for e, v in adjacency[current] if reachable[v] and not visited[v] and e not in blocked
        ]
        if not candidates:
            # Dead end (e.g. every onward node already visited by this ant): back up one
            # step and forbid the edge that led here, rather than discarding the whole tour.
            if len(path_indices) == 1:
                return None
            dead_end_node = path_indices.pop()
            dead_end_edge = edge_ids.pop()
            visited[dead_end_node] = 0
            blocked_from.setdefault(path_indices[-1], set()).add(dead_end_edge)
            steps += 1
            continue
        chosen = _choose_scored(
            candidates, estimate, pheromone_list, remaining_list[current], config.alpha, config.beta,
            rng, wg, pheromone, remaining,
        )
        next_node = next(v for e, v in adjacency[current] if e == chosen)
        path_indices.append(next_node)
        edge_ids.append(chosen)
        visited[next_node] = 1
        steps += 1
    if path_indices[-1] == destination_idx:
        return (path_indices, edge_ids)
    return None


def ant_colony_shortest_path(
    wg: WeightedGraph, origin: int, destination: int, config: AntColonyConfig = AntColonyConfig()
) -> RouteResult:
    """
    Route between two road-graph node ids (OSM ids, not indices). Every distinct tour any ant
    completes is kept, ranked by dynamic_cost; raises ValueError on invalid config, an
    unreachable destination, or when no ant completes a tour.
    """
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

    remaining = remaining_cost_to(wg, destination_idx)
    if not np.isfinite(remaining[origin_idx]):
        raise ValueError(f"destination {destination} is not reachable from origin {origin}")

    max_steps = config.max_steps if config.max_steps is not None else wg.n_nodes - 1
    rng = np.random.default_rng(config.seed)
    pheromone = np.full(wg.n_edges, config.initial_pheromone, dtype=np.float64)
    # Fixed for the colony; pheromone changes only between iterations.
    colony = _ColonyInputs.build(wg, remaining)
    found: Dict[Tuple[int, ...], PathMetrics] = {}
    for _ in range(config.n_iterations):
        pheromone_list = pheromone.tolist()
        tours: List[Tuple[PathMetrics, List[int]]] = []
        for _ in range(config.n_ants):
            result = _build_tour(
                wg, origin_idx, destination_idx, pheromone, remaining, config, max_steps, rng,
                colony, pheromone_list,
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
    """Chain one colony run per consecutive pair of stops into a single route (no alternatives)."""
    if len(stops) < 2:
        raise ValueError("multi_stop_route needs at least an origin and a destination")
    full_path: List[int] = [int(stops[0])]
    for leg_origin, leg_destination in zip(stops, stops[1:]):
        leg = ant_colony_shortest_path(wg, leg_origin, leg_destination, config)
        full_path.extend(leg.best.nodes[1:])
    best = wg.evaluate_path(full_path)
    return RouteResult(best=best, alternatives=[])


# Two routes sharing more than this fraction of the shorter one's length are the same
# route with a small detour, not an alternative worth offering.
MAX_SHARED_FRACTION = 0.7

# Each round, edges already used by a found route cost this much more, compounding, so
# the colony is pushed onto different roads until it finds a genuinely separate route.
REUSE_PENALTY = 1.5


def _path_edges(wg: WeightedGraph, nodes: Sequence[int]) -> List[int]:
    indices = [wg.index_of(n) for n in nodes]
    return [wg.edge_id(u, v) for u, v in zip(indices, indices[1:])]


def shared_fraction(wg: WeightedGraph, a: PathMetrics, b: PathMetrics) -> float:
    """Length the two routes have in common, as a fraction of the shorter one."""
    edges_a, edges_b = set(_path_edges(wg, a.nodes)), set(_path_edges(wg, b.nodes))
    shared = float(wg.distance[list(edges_a & edges_b)].sum()) if edges_a & edges_b else 0.0
    return shared / min(a.distance_m, b.distance_m)


def diverse_routes(
    wg: WeightedGraph,
    stops: Sequence[int],
    k: int = 3,
    config: AntColonyConfig = AntColonyConfig(),
    max_rounds: int = 4,
) -> List[PathMetrics]:
    """
    Up to `k` genuinely different routes through `stops`, cheapest first under `wg`'s
    dynamic cost (the k shortest routes, in the model's own sense of "short").

    The colony's own alternatives are the other tours its ants happened to walk, and
    once pheromone converges those are the best route with a one-node detour -- not
    something to offer a driver. So this uses the penalty method: run the colony, make
    every edge of the routes found so far REUSE_PENALTY times more expensive, run it
    again, and keep a route only if it shares at most MAX_SHARED_FRACTION of its length
    with each route already kept. Every route is scored on the real, unpenalised `wg`.
    Fewer than `k` come back when the network offers no more separate roads.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")
    kept: List[PathMetrics] = []
    searched = wg
    for _ in range(max_rounds):
        if len(stops) == 2:
            result = ant_colony_shortest_path(searched, stops[0], stops[1], config)
            candidates = [result.best] + result.alternatives
        else:
            candidates = [multi_stop_route(searched, stops, config).best]
        for candidate in candidates:
            route = wg.evaluate_path(candidate.nodes)
            if all(shared_fraction(wg, route, other) <= MAX_SHARED_FRACTION for other in kept):
                kept.append(route)
            if len(kept) == k:
                return sorted(kept, key=lambda m: m.dynamic_cost)
        weight = searched.weight.copy()
        for route in kept:
            weight[_path_edges(wg, route.nodes)] *= REUSE_PENALTY
        searched = replace(searched, weight=weight)
    return sorted(kept, key=lambda m: m.dynamic_cost)