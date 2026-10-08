"""
Dynamic Weight Engine: turns the per-edge Congestion Risk Score into a routing cost.

It sits between the MLP decoder (scripts/predict_congestion_risk.py -> risk_edges.csv)
and the ACO routing engine, and is the only place the risk penalty is applied.

    W(u, v) = dist(u, v) * (1 + lambda * Risk(u, v))

dist is the OSM road length in metres carried by the adjacency matrix, Risk is the
decoder's score in [0, 1], and lambda scales how risk-averse the routing is
(lambda = 0 reproduces the distance-only static baseline).

Route Risk Exposure of a path, used to compare routing strategies, is the plain sum of
the risks of its edges - the risk is NOT re-weighted by distance here:

    Risk(P) = sum of Risk(i, j) over the edges (i, j) of P

Both follow Xue et al. (2026), equations 3 and 5.

Edges keep the direction of the road network, so W(u, v) and W(v, u) may differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.graph_data import GraphData

# Risk sensitivity. 2.0 makes a fully congested edge cost three times its length,
# the setting worked through in the methodology.
DEFAULT_LAMBDA = 2.0

# Risk given to an edge the decoder produced no score for, by the plain risk_vector. 0.0
# leaves such an edge at its physical length. Fine on the camera subgraph, where nearly every
# edge is scored (scripts/evaluate_routing.py); on the whole city it makes roads without a
# prediction look empty, so the live app fills them with fill_unscored_risk instead.
# Always read `WeightedGraph.coverage` before trusting a route.
DEFAULT_MISSING_RISK = 0.0


# ---------------------------------------------------------------------------
# core formula
# ---------------------------------------------------------------------------
def dynamic_weight(
    distance: np.ndarray,
    risk: np.ndarray,
    lam: float = DEFAULT_LAMBDA,
) -> np.ndarray:
    """W = dist * (1 + lambda * Risk), elementwise. The single definition of the cost."""
    distance = np.asarray(distance, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)

    if distance.shape != risk.shape:
        raise ValueError(f"distance {distance.shape} and risk {risk.shape} must match")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError(f"lambda must be finite and >= 0, received {lam}")
    if not np.all(np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError("every edge distance must be finite and > 0")
    if not np.all(np.isfinite(risk)) or np.any((risk < 0) | (risk > 1)):
        raise ValueError("every risk must be finite and within [0, 1]")

    return distance * (1.0 + lam * risk)


# ---------------------------------------------------------------------------
# edge distances
# ---------------------------------------------------------------------------
def edge_distances(graph: GraphData) -> np.ndarray:
    """
    Road length in metres for each column of `graph.edge_index`.

    `spatial_topology.build_adjacency` stores `length` as the matrix value, and
    `edge_index_from_adjacency` reads the same matrix through `tocoo()`, so the two line
    up column for column. That is asserted rather than assumed.
    """
    coo = graph.adjacency.tocoo()
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()

    if coo.nnz != src.size:
        raise ValueError(
            f"adjacency has {coo.nnz} stored edges but edge_index has {src.size} columns"
        )
    if not (np.array_equal(coo.row, src) and np.array_equal(coo.col, dst)):
        raise ValueError(
            "adjacency and edge_index disagree on edge order; distances would be "
            "attached to the wrong edges"
        )

    distance = coo.data.astype(np.float64)
    if np.any(distance <= 0) or not np.all(np.isfinite(distance)):
        raise ValueError("the adjacency matrix holds a non-positive or non-finite length")
    return distance


# ---------------------------------------------------------------------------
# weighted graph
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathMetrics:
    """What a route costs. `risk_exposure` is the metric reported against the baseline."""

    nodes: List[int]           # road-network node ids, origin first
    n_edges: int
    distance_m: float          # total physical distance
    risk_exposure: float       # sum of edge risks (RADR eq. 5)
    dynamic_cost: float        # sum of W, what the router minimises
    mean_risk: float           # risk_exposure / n_edges, comparable across route lengths


@dataclass
class WeightedGraph:
    """
    The risk-penalised road graph handed to the ACO routing engine.

    All per-edge arrays are indexed by edge id, i.e. the column order of `edge_index`.
    """

    node_ids: np.ndarray           # [N] road-network node id per graph index
    src: np.ndarray                # [E] graph index of each edge's tail
    dst: np.ndarray                # [E] graph index of each edge's head
    distance: np.ndarray           # [E] metres
    risk: np.ndarray               # [E] congestion risk in [0, 1]
    weight: np.ndarray             # [E] W = dist * (1 + lambda * risk)
    lam: float
    coverage: float                # fraction of edges that got a predicted risk
    scored_edges: int              # how many edges that was
    window: Optional[str] = None   # window this snapshot describes, for traceability

    def __post_init__(self) -> None:
        self._index_of_node: Dict[int, int] = {
            int(n): i for i, n in enumerate(self.node_ids)
        }
        # Outgoing edges per node, as a CSR-style (indptr, edge id) pair so the router can
        # read a node's choices without scanning all E edges.
        order = np.argsort(self.src, kind="stable")
        self._out_edges = order.astype(np.int64)
        self._out_start = np.searchsorted(self.src[order], np.arange(self.n_nodes + 1))

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.weight)

    def index_of(self, node_id: int) -> int:
        """Graph index of a road-network node id."""
        try:
            return self._index_of_node[int(node_id)]
        except KeyError:
            raise KeyError(f"node {node_id} is not in this subgraph") from None

    def out_edges(self, node: int) -> np.ndarray:
        """Edge ids leaving a graph index — the candidate moves for one ant."""
        return self._out_edges[self._out_start[node] : self._out_start[node + 1]]

    def edge_id(self, u: int, v: int) -> int:
        """Edge id of the arc u -> v, both graph indices."""
        for e in self.out_edges(u):
            if self.dst[e] == v:
                return int(e)
        raise KeyError(f"no edge {self.node_ids[u]} -> {self.node_ids[v]}")

    def with_lambda(self, lam: float) -> "WeightedGraph":
        """The same risk snapshot at another risk sensitivity. lam=0 gives the static graph."""
        return WeightedGraph(
            node_ids=self.node_ids,
            src=self.src,
            dst=self.dst,
            distance=self.distance,
            risk=self.risk,
            weight=dynamic_weight(self.distance, self.risk, lam),
            lam=float(lam),
            coverage=self.coverage,
            scored_edges=self.scored_edges,
            window=self.window,
        )

    def evaluate_path(self, path: Sequence[int]) -> PathMetrics:
        """
        Score a route given as road-network node ids.

        Raises if two consecutive nodes are not joined by an arc, so an infeasible route
        can never be reported as a cheap one.
        """
        if len(path) < 2:
            raise ValueError("a path needs at least two nodes")

        indices = [self.index_of(n) for n in path]
        edges = [self.edge_id(u, v) for u, v in zip(indices, indices[1:])]

        risk_exposure = float(self.risk[edges].sum())
        return PathMetrics(
            nodes=[int(n) for n in path],
            n_edges=len(edges),
            distance_m=float(self.distance[edges].sum()),
            risk_exposure=risk_exposure,
            dynamic_cost=float(self.weight[edges].sum()),
            mean_risk=risk_exposure / len(edges),
        )

    def to_networkx(self):
        """DiGraph with `weight` (routing cost), `distance` and `risk` on every edge."""
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(int(n) for n in self.node_ids)
        g.add_edges_from(
            (
                int(self.node_ids[u]),
                int(self.node_ids[v]),
                {"weight": float(w), "distance": float(d), "risk": float(r)},
            )
            for u, v, w, d, r in zip(
                self.src, self.dst, self.weight, self.distance, self.risk
            )
        )
        return g

    def to_dataframe(self) -> pd.DataFrame:
        """One row per edge, for inspection and for the results appendices."""
        return pd.DataFrame(
            {
                "source_node_id": self.node_ids[self.src],
                "target_node_id": self.node_ids[self.dst],
                "distance_m": self.distance,
                "risk": self.risk,
                "dynamic_weight": self.weight,
            }
        )


def build_weighted_graph(
    graph: GraphData,
    risk: np.ndarray,
    lam: float = DEFAULT_LAMBDA,
    coverage: float = 1.0,
    scored_edges: Optional[int] = None,
    window: Optional[str] = None,
) -> WeightedGraph:
    """Apply the dynamic weight to every edge of `graph` for one risk snapshot."""
    risk = np.asarray(risk, dtype=np.float64)
    n_edges = graph.edge_index.shape[1]
    if risk.shape != (n_edges,):
        raise ValueError(f"risk must have shape ({n_edges},), received {risk.shape}")

    distance = edge_distances(graph)
    return WeightedGraph(
        node_ids=np.asarray(graph.node_ids),
        src=graph.edge_index[0].numpy(),
        dst=graph.edge_index[1].numpy(),
        distance=distance,
        risk=risk,
        weight=dynamic_weight(distance, risk, lam),
        lam=float(lam),
        coverage=float(coverage),
        scored_edges=n_edges if scored_edges is None else int(scored_edges),
        window=window,
    )


# ---------------------------------------------------------------------------
# risk_edges.csv -> per-edge risk
# ---------------------------------------------------------------------------
RISK_KEY = ["source_node_id", "target_node_id"]


def load_risk_edges(path: Path) -> pd.DataFrame:
    """Read the decoder's risk_edges.csv and check the columns the engine relies on."""
    frame = pd.read_csv(path)
    missing = {"source_node_id", "target_node_id", "risk"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    if frame["risk"].isna().any():
        raise ValueError(f"{path} has empty risk values")
    return frame


def window_key(frame: pd.DataFrame) -> pd.Series:
    """
    One label per prediction window.

    predict_congestion_risk.py writes window_start; predict_risk.py --edges writes only
    window_end, so fall back to whichever is present.
    """
    for column in ("window_start", "window_end"):
        if column in frame.columns:
            return frame[column].astype(str)
    raise ValueError("risk_edges.csv has neither window_start nor window_end")


def risk_vector(
    graph: GraphData,
    frame: pd.DataFrame,
    missing_risk: float = DEFAULT_MISSING_RISK,
) -> Tuple[np.ndarray, int]:
    """
    Align one window's rows to `graph.edge_index`, by node id pair rather than row order.

    Returns the per-edge risk and how many edges the decoder actually scored.
    Duplicate rows for one edge are averaged; unscored edges fall back to `missing_risk`.
    """
    if not 0.0 <= missing_risk <= 1.0:
        raise ValueError(f"missing_risk must be within [0, 1], received {missing_risk}")

    node_ids = np.asarray(graph.node_ids)
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()

    scores = (
        frame.groupby(RISK_KEY, sort=False)["risk"].mean().astype(np.float64).to_dict()
    )
    risk = np.full(src.size, float(missing_risk), dtype=np.float64)
    scored = 0
    for e, (u, v) in enumerate(zip(node_ids[src], node_ids[dst])):
        value = scores.get((int(u), int(v)))
        if value is not None:
            risk[e] = value
            scored += 1

    out_of_range = (risk < 0) | (risk > 1)
    if out_of_range.any():
        raise ValueError(
            f"{int(out_of_range.sum())} edge(s) carry a risk outside [0, 1]; the decoder "
            "is sigmoid-bounded, so check the risk file"
        )
    return risk, scored


def scored_mask(graph: GraphData, frame: pd.DataFrame) -> np.ndarray:
    """[E] True where `frame` (one window of risk_edges.csv) scores that edge."""
    node_ids = np.asarray(graph.node_ids)
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    keys = set(map(tuple, frame[RISK_KEY].astype(np.int64).to_numpy().tolist()))
    return np.fromiter(((int(u), int(v)) in keys for u, v in zip(node_ids[src], node_ids[dst])),
                       dtype=bool, count=src.size)


# How far, by road, an unscored edge may be from a scored one and still borrow its risk.
# Congestion spills along a corridor for a few blocks; past ~1.5 km the nearest scored road
# says nothing specific about this one, so the window's median stands in instead.
MAX_FILL_M = 1500.0


@dataclass
class NearestScored:
    """
    For every node, the nearest node touching a scored edge (by road distance) and how far.

    Depends only on which edges are scored -- the camera subgraph, the same in every window
    of a risk_edges.csv -- so it is built once and reused for every window.
    """

    source: np.ndarray       # [N] index of the nearest scored node (-9999 when unreachable)
    distance_m: np.ndarray   # [N] road distance to it (inf when unreachable)

    @classmethod
    def build(cls, graph: GraphData, scored: np.ndarray) -> "NearestScored":
        import scipy.sparse as sp
        from scipy.sparse.csgraph import dijkstra

        src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
        n = graph.n_nodes
        seeds = np.unique(np.concatenate([src[scored], dst[scored]]))
        if seeds.size == 0:
            return cls(np.full(n, -9999), np.full(n, np.inf))
        roads = sp.csr_matrix((edge_distances(graph), (src, dst)), shape=(n, n))
        dist, _, source = dijkstra(roads, directed=False, indices=seeds, min_only=True,
                                   return_predecessors=True)
        return cls(np.asarray(source), np.asarray(dist))


def fill_unscored_risk(
    graph: GraphData,
    risk: np.ndarray,
    scored: np.ndarray,
    nearest: NearestScored,
    max_m: float = MAX_FILL_M,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Risk for the edges the decoder did not score, instead of a flat DEFAULT_MISSING_RISK.

    A flat 0 makes every road without a prediction look free of congestion, so the router
    prefers roads with *no data* over roads with data -- on Metro Manila, where the decoder
    scores ~2% of edges around the cameras, ANTROUTE steered off EDSA onto side streets for
    exactly that reason. Instead an unscored edge within `max_m` of a scored road (by road
    distance, from its nearer endpoint) takes that road's risk for this window, and anything
    farther takes the window's median: unknown is treated as typical, not as empty.

    Returns the filled risk and how many edges each rule covered.
    """
    risk = np.asarray(risk, dtype=np.float64).copy()
    scored = np.asarray(scored, dtype=bool)
    if not scored.any():
        return risk, {"scored": 0, "nearest": 0, "median": 0}
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    n = graph.n_nodes

    # risk at each scored node: the mean of the scored edges touching it
    total = np.bincount(src[scored], weights=risk[scored], minlength=n) + np.bincount(
        dst[scored], weights=risk[scored], minlength=n)
    count = np.bincount(src[scored], minlength=n) + np.bincount(dst[scored], minlength=n)
    node_risk = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    reachable = nearest.source >= 0
    borrowed = np.full(n, np.nan)
    borrowed[reachable] = node_risk[nearest.source[reachable]]
    # each unscored edge borrows through whichever endpoint is nearer a scored road
    use_src = nearest.distance_m[src] <= nearest.distance_m[dst]
    edge_value = np.where(use_src, borrowed[src], borrowed[dst])
    edge_distance = np.minimum(nearest.distance_m[src], nearest.distance_m[dst])

    median = float(np.median(risk[scored]))
    missing = ~scored
    near = missing & (edge_distance <= max_m) & np.isfinite(edge_value)
    risk[near] = edge_value[near]
    risk[missing & ~near] = median
    return risk, {"scored": int(scored.sum()), "nearest": int(near.sum()), "median": int((missing & ~near).sum())}


def weighted_graph_from_risk_file(
    graph: GraphData,
    path: Path,
    window: Optional[str] = None,
    lam: float = DEFAULT_LAMBDA,
    missing_risk: float = DEFAULT_MISSING_RISK,
) -> WeightedGraph:
    """Weighted graph for one window of a risk_edges.csv. Latest window if none is given."""
    frame = load_risk_edges(path)
    keys = window_key(frame)
    chosen = window if window is not None else str(keys.max())

    rows = frame[keys == chosen]
    if rows.empty:
        raise ValueError(
            f"no rows for window {chosen!r} in {path}; available: "
            f"{sorted(keys.unique())[:3]} ... ({keys.nunique()} windows)"
        )

    risk, scored = risk_vector(graph, rows, missing_risk)
    return build_weighted_graph(
        graph,
        risk,
        lam=lam,
        coverage=scored / graph.edge_index.shape[1],
        scored_edges=scored,
        window=chosen,
    )


def iter_windows(
    graph: GraphData,
    path: Path,
    lam: float = DEFAULT_LAMBDA,
    missing_risk: float = DEFAULT_MISSING_RISK,
) -> Iterator[WeightedGraph]:
    """Every window of a risk file in chronological order, for batch routing experiments."""
    frame = load_risk_edges(path)
    keys = window_key(frame)
    n_edges = graph.edge_index.shape[1]

    for chosen, rows in frame.groupby(keys, sort=True):
        risk, scored = risk_vector(graph, rows, missing_risk)
        yield build_weighted_graph(
            graph,
            risk,
            lam=lam,
            coverage=scored / n_edges,
            scored_edges=scored,
            window=str(chosen),
        )
