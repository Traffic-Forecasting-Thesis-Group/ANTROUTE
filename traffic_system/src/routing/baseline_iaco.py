"""
===============================================================================
NOT ON THE ARCHITECTURE DIAGRAM -- THIS IS THE COMPARISON BASELINE
===============================================================================
Read this before confusing it with aco_routing.py.

    src/routing/aco_routing.py   = ANTROUTE's PROPOSED method (blocks 19-20).
                                   Routes on PREDICTED risk from the model.
    src/routing/baseline_iaco.py = THIS FILE. The published method we are
                                   measured against. Routes on OBSERVED
                                   traffic, never on our predictions.

    Both are ant colony algorithms, which is exactly why they are easy to mix
    up. The difference that matters for the thesis is the input: ours uses the
    Congestion Risk Score (block 17); this one uses recorded congestion labels
    and YOLO vehicle counts.

    Evaluation result (50 trips, outputs/routing_eval/): ANTROUTE 83.66% route
    optimality vs this baseline's 73.60%, Wilcoxon p = 4.4e-6. The baseline
    wins on ETA accuracy.

===============================================================================

Baseline router: Improved Ant Colony Algorithm (IACO) from Cheng (2023),
"Dynamic Path Optimization Based on Improved Ant Colony Algorithm",
J. Adv. Transportation, doi:10.1155/2023/7651100.

Implemented as written in the paper, separate from ANTROUTE's aco_routing.py:
  1. Combined cost (eq. 4-6): s = 0.637*length + 0.258*time + 0.105*flow,
     each term divided by its maximum (eq. 5), weights from AHP (Table 3).
  2. Directional guidance (eq. 8): eta = 1 / (s_ij + d_jD), d_jD = straight-line
     distance from node j to the destination.
  3. Transfer rule (eq. 10-11): with probability q0 take the best edge,
     otherwise pick randomly in proportion to tau^alpha * eta^beta.
  4. Pheromone updates: local (eq. 12-13) and global (eq. 14).
  5. Dynamic re-planning (eq. 9): on new traffic data, tau *= t_new / t_old
     and the route is searched again from the vehicle's current node.

Inputs are observed traffic (not predicted): travel time from the congestion
labels, flow from YOLO vehicle counts.

Deviations from the paper:
  - 20 ants x 60 iterations (same as ANTROUTE) instead of 1.5 x n_nodes ants.
  - Global evaporation xi is not given in the paper; uses 0.3 (same as rho).
  - Dead-end recovery (dead_end_recovery=True, the default). The paper's test networks have
    no dead ends; Metro Manila's do, and under the literal tabu rule every ant died there
    (28,800 of 28,800 in a 24-trip test), so the "baseline" was really shortest distance.
    Two feasibility-only changes keep the algorithm itself intact: an ant never takes a road
    from which the destination cannot be reached at all (pure topology, no cost or heuristic
    information), and an ant left with no unvisited road steps back one node and forbids the
    road it came in on instead of dying. The cost (eq. 6), heuristic (eq. 8), transfer rule
    (eq. 10-11) and pheromone updates (eq. 12-14) are unchanged. dead_end_recovery=False is
    the literal paper behaviour.
  - Eq. 9 is literal, so it raises pheromone on edges that got slower.
    invert_eq9 and reset_on_replan are sensitivity variants.

Validated in tests/test_baseline_iaco.py: reproduces the paper's AHP values and
Table 8. The paper's reported optimal path ranks 5th of 15 under its own cost.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, dijkstra

# Table 3: path length (C1), travel time (C2), traffic flow (C3), Saaty scale.
AHP_JUDGMENT = np.array(
    [
        [1.0, 3.0, 5.0],
        [1.0 / 3.0, 1.0, 3.0],
        [1.0 / 5.0, 1.0 / 3.0, 1.0],
    ]
)

# Table 2: average random consistency index by matrix order.
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# AHP weights (sec. 3.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AhpWeights:
    weights: np.ndarray     # normalised eigenvector, sums to 1
    lambda_max: float
    ci: float               # consistency index, (lambda_max - n) / (n - 1)
    cr: float               # consistency ratio, CI / RI; must be < 0.1

    @property
    def consistent(self) -> bool:
        return self.cr < 0.1


def ahp_weights(matrix: np.ndarray = AHP_JUDGMENT) -> AhpWeights:
    """Hierarchical single arrangement, Steps 1-4, and the consistency test (eq. 4)."""
    a = np.asarray(matrix, dtype=np.float64)
    n = a.shape[0]
    if a.shape != (n, n) or n < 1:
        raise ValueError(f"judgment matrix must be square, received {a.shape}")
    if np.any(a <= 0) or not np.allclose(a * a.T, 1.0):
        raise ValueError("judgment matrix must be positive and reciprocal (a_ji = 1 / a_ij)")
    if n not in RANDOM_INDEX:
        raise ValueError(f"no random consistency index for order {n}")

    m = np.prod(a, axis=1)                      # Step 1
    root = m ** (1.0 / n)                       # Step 2
    w = root / root.sum()                       # Step 3
    lambda_max = float(np.sum((a @ w) / (n * w)))   # Step 4
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX[n]
    cr = ci / ri if ri > 0 else 0.0
    return AhpWeights(weights=w, lambda_max=lambda_max, ci=float(ci), cr=float(cr))


# ---------------------------------------------------------------------------
# comprehensive cost (sec. 3.3)
# ---------------------------------------------------------------------------
def extremum_normalise(x: np.ndarray) -> np.ndarray:
    """Eq. 5: x' = x / max(x), mapping non-negative values into [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("values to normalise must be finite and non-negative")
    top = x.max() if x.size else 0.0
    return x / top if top > 0 else np.zeros_like(x)


def comprehensive_cost(
    distance: np.ndarray,
    travel_time: np.ndarray,
    flow: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Eq. 6: s_ij = w1 * d_ij + w2 * t_ij + w3 * n_ij, each term extremum-normalised."""
    w = ahp_weights().weights if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (3,):
        raise ValueError(f"need three criterion weights, received {w.shape}")
    distance, travel_time, flow = (np.asarray(v, dtype=np.float64) for v in (distance, travel_time, flow))
    if not (distance.shape == travel_time.shape == flow.shape):
        raise ValueError("distance, travel_time and flow must have the same shape")
    if np.any(distance <= 0):
        raise ValueError("every edge distance must be > 0")
    return (
        w[0] * extremum_normalise(distance)
        + w[1] * extremum_normalise(travel_time)
        + w[2] * extremum_normalise(flow)
    )


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))


# ---------------------------------------------------------------------------
# camera observations -> every edge
# ---------------------------------------------------------------------------
def fill_from_nearest_camera(
    src: np.ndarray,
    dst: np.ndarray,
    n_nodes: int,
    camera_values: Mapping[int, float],
) -> np.ndarray:
    """
    Spread camera readings to every edge: each edge takes the value of its nearest
    camera (by hop count; ties are averaged). Edges with no camera in reach get the
    mean of all cameras.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    observed = {int(c): float(v) for c, v in camera_values.items() if v is not None and np.isfinite(v)}
    if not observed:
        raise ValueError("no camera has an observation to fill from")

    neighbours: List[List[int]] = [[] for _ in range(n_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        neighbours[u].append(v)
        neighbours[v].append(u)

    hops = np.full(n_nodes, -1, dtype=np.int64)
    total = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    queue: deque = deque()
    for c, v in observed.items():
        hops[c] = 0
        total[c] = v
        count[c] = 1
        queue.append(c)
    # BFS from all cameras at once; a node equally near two cameras gets their mean.
    while queue:
        u = queue.popleft()
        for v in neighbours[u]:
            if hops[v] == -1:
                hops[v] = hops[u] + 1
                total[v] = total[u]
                count[v] = count[u]
                queue.append(v)
            elif hops[v] == hops[u] + 1:
                total[v] += total[u]
                count[v] += count[u]

    fallback = float(np.mean(list(observed.values())))
    node_value = np.where(count > 0, total / np.maximum(count, 1), fallback)
    node_hops = np.where(hops >= 0, hops, np.iinfo(np.int64).max)

    hs, hd = node_hops[src], node_hops[dst]
    vs, vd = node_value[src], node_value[dst]
    return np.where(hs < hd, vs, np.where(hd < hs, vd, (vs + vd) / 2.0))


# ---------------------------------------------------------------------------
# the road graph as the baseline sees it
# ---------------------------------------------------------------------------
@dataclass
class IacoGraph:
    """Road graph with node coordinates."""

    node_ids: np.ndarray       # [N] road-network node id per graph index
    src: np.ndarray            # [E]
    dst: np.ndarray            # [E]
    distance: np.ndarray       # [E] metres
    lat: np.ndarray            # [N]
    lon: np.ndarray            # [N]

    def __post_init__(self) -> None:
        self._index_of_node = {int(n): i for i, n in enumerate(self.node_ids)}
        order = np.argsort(self.src, kind="stable")
        self._out_edges = order.astype(np.int64)
        self._out_start = np.searchsorted(self.src[order], np.arange(self.n_nodes + 1))
        self._adjacency: Optional[List[List[Tuple[int, int]]]] = None

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def index_of(self, node_id: int) -> int:
        try:
            return self._index_of_node[int(node_id)]
        except KeyError:
            raise KeyError(f"node {node_id} is not in this subgraph") from None

    def out_edges(self, node: int) -> np.ndarray:
        return self._out_edges[self._out_start[node] : self._out_start[node + 1]]

    def adjacency(self, reachable: Optional[np.ndarray] = None) -> List[List[Tuple[int, int]]]:
        """
        (edge, next node) for the roads out of each node, in out_edges order, as plain ints
        for the ants' inner loop. With `reachable`, only roads into reachable nodes.
        """
        if self._adjacency is None:
            edges = self._out_edges.tolist()
            heads = self.dst[self._out_edges].tolist()
            starts = self._out_start.tolist()
            self._adjacency = [list(zip(edges[a:b], heads[a:b])) for a, b in zip(starts, starts[1:])]
        if reachable is None:
            return self._adjacency
        ok = reachable.tolist()
        return [[(e, v) for e, v in row if ok[v]] for row in self._adjacency]

    def edges_of(self, path_indices: Sequence[int]) -> List[int]:
        edges = []
        for u, v in zip(path_indices, path_indices[1:]):
            match = [e for e in self.out_edges(u) if self.dst[e] == v]
            if not match:
                raise KeyError(f"no edge {self.node_ids[u]} -> {self.node_ids[v]}")
            edges.append(int(match[0]))
        return edges

    def can_reach(self, destination_idx: int) -> np.ndarray:
        """[N] True where some directed road path leads to the destination (topology only)."""
        reversed_graph = sp.csr_matrix(
            (np.ones(self.n_edges), (self.dst, self.src)), shape=(self.n_nodes, self.n_nodes)
        )
        reachable = np.zeros(self.n_nodes, dtype=bool)
        reachable[breadth_first_order(reversed_graph, destination_idx, directed=True, return_predecessors=False)] = True
        return reachable

    def goal_distance(self, destination_idx: int) -> np.ndarray:
        """d_jD (eq. 8): normalised straight-line distance from each node to the destination."""
        d = haversine_m(self.lat, self.lon, self.lat[destination_idx], self.lon[destination_idx])
        return extremum_normalise(d)


def corridor_subgraph(
    g: IacoGraph,
    origin_idx: int,
    destination_idx: int,
    margin: float,
    buffer_m: float = 500.0,
) -> Tuple[IacoGraph, np.ndarray]:
    """
    The roads inside an ellipse around a trip: nodes whose straight-line distance to the
    origin plus to the destination is at most (1 + margin) x the trip's straight-line
    length + buffer_m. Returns the subgraph and, per subgraph edge, its edge index in g.

    The paper's test networks are a few dozen nodes; on all of Metro Manila the
    goal-distance term (eq. 8), normalised over every node in the city, barely separates
    nearby nodes, so ants wander for thousands of steps on a trip of a hundred.
    """
    od = haversine_m(g.lat[origin_idx], g.lon[origin_idx], g.lat[destination_idx], g.lon[destination_idx])
    detour = haversine_m(g.lat, g.lon, g.lat[origin_idx], g.lon[origin_idx]) + haversine_m(
        g.lat, g.lon, g.lat[destination_idx], g.lon[destination_idx]
    )
    keep = detour <= od * (1.0 + margin) + buffer_m
    keep[[origin_idx, destination_idx]] = True
    nodes = np.flatnonzero(keep)
    new_index = np.full(g.n_nodes, -1, dtype=np.int64)
    new_index[nodes] = np.arange(len(nodes))
    edges = np.flatnonzero(keep[g.src] & keep[g.dst])
    sub = IacoGraph(
        node_ids=g.node_ids[nodes],
        src=new_index[g.src[edges]],
        dst=new_index[g.dst[edges]],
        distance=g.distance[edges],
        lat=g.lat[nodes],
        lon=g.lon[nodes],
    )
    return sub, edges


def spatial_shortest_path(g: IacoGraph, origin: int, destination: int) -> List[int]:
    """Shortest path by road length only (Dijkstra). Returns node ids."""
    o, d = g.index_of(origin), g.index_of(destination)
    matrix = sp.csr_matrix((g.distance, (g.src, g.dst)), shape=(g.n_nodes, g.n_nodes))
    dist, pred = dijkstra(matrix, directed=True, indices=o, return_predecessors=True)
    if not np.isfinite(dist[d]):
        raise ValueError(f"destination {destination} is not reachable from origin {origin}")
    path = [d]
    while path[-1] != o:
        path.append(int(pred[path[-1]]))
    return [int(g.node_ids[i]) for i in reversed(path)]


# ---------------------------------------------------------------------------
# the colony (sec. 4.2)
# ---------------------------------------------------------------------------
@dataclass
class IacoConfig:
    n_ants: int = 20
    n_iterations: int = 60
    alpha: float = 1.0           # sec. 5.1.2 (3)
    beta: float = 5.0            # sec. 5.1.2 (4)
    rho: float = 0.3             # local volatility, sec. 5.1.2 (5)
    xi: float = 0.3              # global volatility; not given in the paper
    q0: float = 0.7              # large network, sec. 4.2.1
    q: float = 1.0               # pheromone constant Q
    initial_pheromone: float = 1.0
    max_steps: Optional[int] = None
    dead_end_recovery: bool = True  # adaptation for real road networks; False = literal paper
    invert_eq9: bool = False       # variant: tau * t_old / t_new
    reset_on_replan: bool = False  # variant: fresh pheromone on each re-plan
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.n_ants <= 0 or self.n_iterations <= 0:
            raise ValueError("n_ants and n_iterations must be greater than zero")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if not (0 < self.rho < 1 and 0 < self.xi < 1):
            raise ValueError("rho and xi must lie in (0, 1)")
        if not 0 <= self.q0 <= 1:
            raise ValueError("q0 must lie in [0, 1]")
        if self.q <= 0 or self.initial_pheromone <= 0:
            raise ValueError("Q and the initial pheromone must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")


@dataclass
class SearchResult:
    path: List[int]                  # graph indices, start first
    edges: List[int]
    cost: float                      # sum of s along the path
    converged_at: int                # iteration the best path was last improved
    curve: List[float] = field(default_factory=list)   # best-so-far cost per iteration
    stable_from: Optional[int] = None                  # paper's "number of iterations"
    iteration_best: List[float] = field(default_factory=list)  # each iteration's best cost


def _transfer(
    candidates: np.ndarray,
    tau: np.ndarray,
    eta: np.ndarray,
    config: IacoConfig,
    rng: np.random.Generator,
) -> int:
    """Eq. 10-11: choose the next edge."""
    with np.errstate(over="ignore", under="ignore"):
        score = tau[candidates] ** config.alpha * eta ** config.beta
    if rng.random() < config.q0:
        return int(candidates[int(np.argmax(score))])
    total = score.sum()
    if not np.isfinite(total) or total <= 0:
        return int(rng.choice(candidates))
    return int(rng.choice(candidates, p=score / total))


def _transfer_scored(candidates: List[int], scores: List[float], config: IacoConfig, rng: np.random.Generator) -> int:
    """_transfer with tau^alpha * eta^beta already computed: the same rule and random draws."""
    if rng.random() < config.q0:
        # First of the best, as np.argmax. Scores cannot be NaN: tau stays positive.
        return candidates[max(range(len(scores)), key=scores.__getitem__)]
    total = float(np.array(scores).sum())
    if not np.isfinite(total) or total <= 0:
        return int(rng.choice(np.array(candidates, dtype=np.int64)))
    # rng.choice(candidates, p=scores / total), step for step: a cumulative distribution
    # normalised by its last value, searched (right side) with one uniform draw.
    cdf, running = [], 0.0
    for x in scores:
        running += x / total
        cdf.append(running)
    last = cdf[-1]
    cdf = [c / last for c in cdf]
    return candidates[bisect_right(cdf, rng.random())]


def _ant_tour(
    adjacency: List[List[Tuple[int, int]]],
    start: int,
    destination: int,
    score: List[float],
    forbidden: frozenset,
    max_steps: int,
    config: IacoConfig,
    rng: np.random.Generator,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    One ant from start to destination. `adjacency` is g.adjacency(), already limited to
    roads from which the destination can be reached when dead-end recovery is on; `score`
    is tau^alpha * eta^beta (eq. 8, 10) per edge for this iteration.
    """
    current = start
    visited = set(forbidden) | {start}
    path, edges = [start], []
    blocked: Dict[int, set] = {}
    steps = 0
    # A step back does not count toward max_steps: each one forbids a road for the rest of
    # the tour, so there can be at most one per edge.
    while steps < max_steps:
        if current == destination:
            return path, edges
        skip = blocked.get(current, ())
        allowed = [e for e, v in adjacency[current] if v not in visited and e not in skip]
        if not allowed:
            if not config.dead_end_recovery or len(path) == 1:
                return None
            # Dead end: step back one node and forbid the road that led here.
            path.pop()
            dead_edge = edges.pop()
            current = path[-1]
            blocked.setdefault(current, set()).add(dead_edge)
            continue
        chosen = _transfer_scored(allowed, [score[e] for e in allowed], config, rng)
        current = next(v for e, v in adjacency[current] if e == chosen)
        visited.add(current)
        path.append(current)
        edges.append(chosen)
        steps += 1
    return (path, edges) if current == destination else None


def iaco_search(
    g: IacoGraph,
    start: int,
    destination: int,
    s: np.ndarray,
    tau: np.ndarray,
    config: IacoConfig,
    rng: np.random.Generator,
    forbidden: Sequence[int] = (),
) -> Optional[SearchResult]:
    """Steps 1-7 of the paper. Updates tau in place. Returns None if no ant arrives."""
    start, destination = int(start), int(destination)
    goal = g.goal_distance(destination)
    reachable = g.can_reach(destination) if config.dead_end_recovery else None
    adjacency = g.adjacency(reachable)
    max_steps = config.max_steps if config.max_steps is not None else g.n_nodes - 1
    forbidden = frozenset(int(n) for n in forbidden)
    # eta (eq. 8) depends only on the edge and the destination, so it is fixed for the search.
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        eta_beta = (1.0 / (s + goal[g.dst])) ** config.beta

    best: Optional[SearchResult] = None
    curve: List[float] = []
    iteration_best: List[float] = []
    for iteration in range(1, config.n_iterations + 1):
        # tau only changes between iterations, so every ant of this one sees the same scores.
        with np.errstate(over="ignore", under="ignore"):
            score = (tau ** config.alpha * eta_beta).tolist()
        tours = []
        for _ in range(config.n_ants):
            tour = _ant_tour(adjacency, start, destination, score, forbidden, max_steps, config, rng)
            if tour is not None:
                path, edges = tour
                tours.append((path, edges, float(s[edges].sum())))

        # Local update (eq. 12-13).
        tau *= 1.0 - config.rho
        for _, edges, cost in tours:
            tau[edges] += config.q / cost

        if tours:
            path, edges, cost = min(tours, key=lambda t: t[2])
            # Global update (eq. 14): reinforce this iteration's best tour.
            tau *= 1.0 - config.xi
            tau[edges] += config.q / cost
            if best is None or cost < best.cost - 1e-12:
                best = SearchResult(path=path, edges=edges, cost=cost, converged_at=iteration)
        iteration_best.append(min((c for _, _, c in tours), default=float("nan")))
        curve.append(best.cost if best is not None else float("nan"))

    if best is not None:
        best.curve = curve
        best.iteration_best = iteration_best
        best.stable_from = stable_from(iteration_best)
    return best


def stable_from(iteration_best: Sequence[float]) -> Optional[int]:
    """First iteration after which each iteration's best stays the same."""
    if not iteration_best or not np.isfinite(iteration_best[-1]):
        return None
    final = iteration_best[-1]
    i = len(iteration_best)
    while i > 1 and abs(iteration_best[i - 2] - final) <= 1e-12:
        i -= 1
    return i


def rescale_pheromone(tau: np.ndarray, t_old: np.ndarray, t_new: np.ndarray, invert: bool = False) -> None:
    """Eq. 9: tau *= t_new / t_old (in place)."""
    ratio = (t_old / t_new) if invert else (t_new / t_old)
    tau *= ratio


# ---------------------------------------------------------------------------
# time-varying traffic and the dynamic loop (sec. 4.2.3, Steps 8-9)
# ---------------------------------------------------------------------------
@dataclass
class TrafficSnapshots:
    """Observed traffic over time. Each snapshot holds until the next one."""

    times: np.ndarray            # [W] seconds
    travel_time: np.ndarray      # [W, E] seconds
    flow: np.ndarray             # [W, E] vehicles
    congestion: np.ndarray       # [W, E] observed level in [0, 1]

    def __post_init__(self) -> None:
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("snapshot times must be strictly increasing")
        if np.any(self.travel_time <= 0):
            raise ValueError("every travel time must be > 0")

    def index_at(self, t: float) -> int:
        """Index of the latest snapshot at or before t."""
        return max(int(np.searchsorted(self.times, t, side="right")) - 1, 0)


def drive(g: IacoGraph, edges: Sequence[int], departure: float, traffic: TrafficSnapshots) -> Tuple[float, float]:
    """Actual travel time and congestion of a route, using the traffic at the time each edge is driven."""
    clock = departure
    exposure = 0.0
    for e in edges:
        w = traffic.index_at(clock)
        exposure += float(traffic.congestion[w, e])
        clock += float(traffic.travel_time[w, e])
    return clock - departure, exposure


@dataclass
class DynamicTrip:
    path: List[int]              # node ids actually driven
    edges: List[int]
    travel_time_s: float
    congestion_exposure: float
    initial_cost: float          # cost s of the first plan
    converged_at: int
    stable_from: Optional[int]
    replans: int
    curve: List[float]


def iaco_dynamic_trip(
    g: IacoGraph,
    origin: int,
    destination: int,
    departure: float,
    traffic: TrafficSnapshots,
    config: IacoConfig = IacoConfig(),
) -> DynamicTrip:
    """
    Plan, then drive. At each node, if new traffic data has arrived, apply eq. 9 and
    re-plan from that node (Steps 8-9). Nodes already driven are not revisited.
    """
    config.validate()
    o, d = g.index_of(origin), g.index_of(destination)
    if o == d:
        raise ValueError(f"origin {origin} and destination {destination} are the same node")
    rng = np.random.default_rng(config.seed)
    tau = np.full(g.n_edges, config.initial_pheromone, dtype=np.float64)

    def cost_at(w: int) -> np.ndarray:
        return comprehensive_cost(g.distance, traffic.travel_time[w], traffic.flow[w])

    w_plan = traffic.index_at(departure)
    first = iaco_search(g, o, d, cost_at(w_plan), tau, config, rng)
    if first is None:
        raise ValueError(
            f"no ant reached {destination} from {origin} in {config.n_iterations} iterations "
            f"of {config.n_ants} ants"
        )

    plan = list(first.edges)
    driven_nodes, driven_edges = [o], []
    clock, exposure, replans = departure, 0.0, 0
    current = o
    while current != d:
        w_now = traffic.index_at(clock)
        if w_now != w_plan:
            if config.reset_on_replan:
                tau[:] = config.initial_pheromone
            else:
                rescale_pheromone(tau, traffic.travel_time[w_plan], traffic.travel_time[w_now], config.invert_eq9)
            again = iaco_search(g, current, d, cost_at(w_now), tau, config, rng, forbidden=driven_nodes[:-1])
            if again is not None:
                plan = list(again.edges)
                replans += 1
            w_plan = w_now
        e = plan.pop(0)
        exposure += float(traffic.congestion[w_now, e])
        clock += float(traffic.travel_time[w_now, e])
        current = int(g.dst[e])
        driven_nodes.append(current)
        driven_edges.append(e)

    return DynamicTrip(
        path=[int(g.node_ids[i]) for i in driven_nodes],
        edges=driven_edges,
        travel_time_s=clock - departure,
        congestion_exposure=exposure,
        initial_cost=first.cost,
        converged_at=first.converged_at,
        stable_from=first.stable_from,
        replans=replans,
        curve=first.curve,
    )
